// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: full chunked Gated Delta Rule forward for one head, sequential
// over chunks, holding state S [K,V] on-core. Derived from flash-linear-attention
// `naive_chunk_kda`. fp32 / HiFi4 throughout.
//
// Per chunk (C=chunk, K=key dim, V=val dim; Ct=C/32, Kt=K/32, Vt=V/32):
//   v_beta = v*beta ; k_beta = k*beta
//   decay = cumsum(g) = tril @ g ; decay_exp = exp(decay)
//   L_mask = tril( exp(decay_i - decay_j) )
//   N = strictly_lower(k_beta@k^T * L_mask) ; T_inv = (I+N)^-1  (Horner)
//   u = T_inv @ v_beta ; w = T_inv @ (k_beta*decay_exp)
//   intra = (q@k^T) * L_mask
//   q_decay = q*decay_exp ; k_dec_t = transpose(k * exp(decay_last - decay))
//   v_prime = w@S ; v_new = u - v_prime ; o = q_decay@S + intra@v_new
//   S = S*exp(decay_last) + k_dec_t@v_new

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_eye = 5, cb_tril = 6, cb_ones = 7, cb_S = 8;
constexpr uint32_t cb_decay = 9, cb_decay_exp = 10, cb_decayfac = 11;
constexpr uint32_t cb_lmask = 12, cb_Tinv = 13, cb_vbeta = 14, cb_kbeta = 15;
constexpr uint32_t cb_out = 16, cb_u = 17, cb_w = 18, cb_qdecay = 19;
constexpr uint32_t cb_intra = 20, cb_s2 = 21, cb_vnew = 22, cb_ointer = 23;
constexpr uint32_t cb_kdec_t = 24, cb_supd = 25, cb_stmp = 26, cb_final = 27;
constexpr uint32_t cb_scr1 = 28, cb_scr2 = 29, cb_scr3 = 30, cb_s3 = 31;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// ceil(log2(x)); used to size the Neumann-doubling triangular solve.
constexpr uint32_t clog2(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) {
        r++;
    }
    return r;
}

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);  // mixed bf16/fp32 CBs: set packer to this output's format
    // matmul_tiles(a,b): in0=a->srcB, in1=b->srcA. Reconfig unpack src formats to match (the op
    // init only asserts formats, it does not set them), else fp32/bf16 CBs are read at the wrong
    // format and produce garbage.
    reconfig_data_format(b, a);
    matmul_init(a, b, tr ? 1 : 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    if (op == 0) {
        add_tiles_init(a, b);
    } else if (op == 1) {
        sub_tiles_init(a, b);
    } else {
        mul_tiles_init(a, b);
    }
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        if (op == 0) {
            add_tiles(a, b, i, i, 0);
        } else if (op == 1) {
            sub_tiles(a, b, i, i, 0);
        } else {
            mul_tiles(a, b, i, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = copy(in), n tiles.
void cpy(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    copy_tile_to_dst_init_short(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

void expc(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    copy_tile_to_dst_init_short(in);
    exp_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init_short(a, col);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, col, mi * Nt + ni, mi, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// out[Mt,Nt] = A[Mt,Nt] - row[1,Nt]  (broadcast the single row of `row` across M)
void bcast_rows_sub(uint32_t a, uint32_t row, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, row);  // bcast(a,row): a->srcA, row->srcB
    sub_bcast_rows_init_short(a, row);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            sub_tiles_bcast_rows(a, row, mi * Nt + ni, ni, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// out = S * scalar[0,0], n tiles.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);  // bcast(a,scal): a->srcA, scal->srcB
    mul_tiles_bcast_scalar_init_short(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out[1,Ct] row-form = transpose of col[Ct,1]; produces Ct tiles (each row0 = a 32-chunk of col).
void transpose_col(uint32_t in, uint32_t o, uint32_t Ct) {
    cb_reserve_back(o, Ct);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    transpose_init(in);
    for (uint32_t i = 0; i < Ct; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, Ct);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    const uint32_t NC = get_arg_val<uint32_t>(0);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t C = Ct * 32;

    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    // Constants (loaded once by reader). Initial state is in cb_S (reader pushed it).
    WAIT(cb_eye, cc);
    WAIT(cb_tril, cc);
    WAIT(cb_ones, cc);

    for (uint32_t c = 0; c < NC; c++) {
        // Recurrent state uses THREE single-producer CBs so no CB is ever produced by both
        // the reader and compute (that reader->compute->compute producer switch desyncs the
        // CB page pointers and deadlocks the next chunk's WAIT at NC>=3):
        //   cb_S  : reader-produced initial state, consumed only by chunk 0.
        //   cb_s2 / cb_s3 : compute-only ping-pong for chunk outputs (compute->compute).
        // Chunk c reads cur_S, writes nxt_S (or cb_final on the last chunk).
        const uint32_t cur_S = (c == 0) ? cb_S : ((c & 1u) ? cb_s2 : cb_s3);
        const uint32_t nxt_S = (c & 1u) ? cb_s3 : cb_s2;
        // NOTE: cur_S is WAIT'd later (just before the scan), not here — waiting on the
        // state at the chunk start blocks the unpacker before it consumes this chunk's
        // inputs, starving the reader (backpressure deadlock at NC>=3).

        WAIT(cb_q, ck);
        WAIT(cb_k, ck);
        WAIT(cb_v, cv);
        WAIT(cb_g, Ct);
        WAIT(cb_beta, Ct);

        // ---- P1: v_beta, k_beta ----
        bcast_cols_mul(cb_v, cb_beta, cb_vbeta, Ct, Vt);
        WAIT(cb_vbeta, cv);
        bcast_cols_mul(cb_k, cb_beta, cb_kbeta, Ct, Kt);
        WAIT(cb_kbeta, ck);
        POP(cb_beta, Ct);
        POP(cb_v, cv);

        // ---- P2: decay = tril@g, decay_exp, decay_row ----
        mm(cb_tril, cb_g, cb_decay, Ct, Ct, 1, false);
        WAIT(cb_decay, Ct);
        expc(cb_decay, cb_decay_exp, Ct);
        WAIT(cb_decay_exp, Ct);
        transpose_col(cb_decay, cb_scr1, Ct);  // decay_row in scr1
        WAIT(cb_scr1, Ct);

        // ---- L_mask = tril(exp(decay_i - decay_j)) ----
        bcast_cols_mul(cb_ones, cb_decay, cb_scr2, Ct, Ct);  // decay_i everywhere
        WAIT(cb_scr2, cc);
        bcast_rows_sub(cb_scr2, cb_scr1, cb_scr3, Ct, Ct);  // decay_i - decay_j
        WAIT(cb_scr3, cc);
        POP(cb_scr1, Ct);  // decay_row done
        POP(cb_scr2, cc);
        ew(cb_scr3, cb_tril, cb_scr2, cc, 2);  // *tril (zero upper)
        WAIT(cb_scr2, cc);
        POP(cb_scr3, cc);
        expc(cb_scr2, cb_scr3, cc);  // exp
        WAIT(cb_scr3, cc);
        POP(cb_scr2, cc);
        ew(cb_scr3, cb_tril, cb_lmask, cc, 2);  // *tril again -> L_mask
        WAIT(cb_lmask, cc);
        POP(cb_scr3, cc);

        // ---- decayfac = exp(g_sum - decay) ----
        // (dl = exp(g_sum) is recomputed at the scan from decayfac[0]*decay_exp[0] so its CB
        //  slot can be reused as the third ping-pong state buffer cb_s3.)
        mm(cb_ones, cb_g, cb_scr1, Ct, Ct, 1, false);  // g_sum in every row (col form)
        WAIT(cb_scr1, Ct);
        POP(cb_g, Ct);
        ew(cb_scr1, cb_decay, cb_scr2, Ct, 1);  // g_sum - decay
        WAIT(cb_scr2, Ct);
        POP(cb_scr1, Ct);
        POP(cb_decay, Ct);
        expc(cb_scr2, cb_decayfac, Ct);
        WAIT(cb_decayfac, Ct);
        POP(cb_scr2, Ct);

        // ---- N = strictly_lower(k_beta@k^T * L_mask); T_inv = (I - negN)^-1 (Horner) ----
        // negN is strictly-lower (nilpotent: negN^C = 0), so T_inv = sum_{i<C} negN^i, summed by the
        // linear Horner recurrence R = I + negN@R (C-1 steps). NOTE: a Neumann-*doubling* rewrite was
        // tried (~log2(C) steps) but is numerically unsafe here — it forms high matrix powers
        // negN^{2^k} whose large intermediates destroy the result on ill-conditioned chunks (all-nan
        // at NC>=16). Horner keeps every intermediate bounded (~the partial inverse), so it is stable.
        mm(cb_kbeta, cb_k, cb_scr1, Ct, Kt, Ct, true);  // kk = k_beta @ k^T
        WAIT(cb_scr1, cc);
        ew(cb_scr1, cb_lmask, cb_scr2, cc, 2);  // kk_masked = kk * L_mask
        WAIT(cb_scr2, cc);
        POP(cb_scr1, cc);
        ew(cb_scr2, cb_eye, cb_scr1, cc, 2);  // diag(kk_masked)
        WAIT(cb_scr1, cc);
        // negN = diag - kk_masked = -(strictly_lower(kk_masked))  (M, kept in cb_scr3)
        ew(cb_scr1, cb_scr2, cb_scr3, cc, 1);
        WAIT(cb_scr3, cc);
        POP(cb_scr1, cc);
        POP(cb_scr2, cc);

        // R_1 = eye + negN  (negN in cb_scr3)
        ew(cb_eye, cb_scr3, cb_Tinv, cc, 0);
        WAIT(cb_Tinv, cc);
        for (uint32_t m = 2; m < C; m++) {
            mm(cb_scr3, cb_Tinv, cb_scr1, Ct, Ct, Ct, false);  // NR = negN @ R
            WAIT(cb_scr1, cc);
            POP(cb_Tinv, cc);
            ew(cb_eye, cb_scr1, cb_Tinv, cc, 0);  // R = eye + NR
            WAIT(cb_Tinv, cc);
            POP(cb_scr1, cc);
        }
        POP(cb_scr3, cc);  // negN done

        // ---- u = T_inv @ v_beta, w = T_inv @ (k_beta*decay_exp) ----
        mm(cb_Tinv, cb_vbeta, cb_u, Ct, Ct, Vt, false);
        WAIT(cb_u, cv);
        POP(cb_vbeta, cv);
        bcast_cols_mul(cb_kbeta, cb_decay_exp, cb_scr1, Ct, Kt);  // k_beta * decay_exp
        WAIT(cb_scr1, ck);
        POP(cb_kbeta, ck);
        mm(cb_Tinv, cb_scr1, cb_w, Ct, Ct, Kt, false);
        WAIT(cb_w, ck);
        POP(cb_scr1, ck);
        POP(cb_Tinv, cc);

        // ---- intra = (q@k^T) * L_mask ; q_decay = q*decay_exp ; k_dec_t ----
        mm(cb_q, cb_k, cb_scr1, Ct, Kt, Ct, true);  // qk = q @ k^T
        WAIT(cb_scr1, cc);
        ew(cb_scr1, cb_lmask, cb_intra, cc, 2);
        WAIT(cb_intra, cc);
        POP(cb_scr1, cc);
        POP(cb_lmask, cc);
        bcast_cols_mul(cb_q, cb_decay_exp, cb_qdecay, Ct, Kt);
        WAIT(cb_qdecay, ck);
        POP(cb_q, ck);
        // decay_exp kept alive: reused at the scan to recompute dl = exp(g_sum).
        bcast_cols_mul(cb_k, cb_decayfac, cb_scr1, Ct, Kt);  // k * exp(decay_last-decay)
        WAIT(cb_scr1, ck);
        POP(cb_k, ck);
        // decayfac kept alive: reused at the scan to recompute dl = exp(g_sum).
        // k_dec_t = transpose(k_dec) [K,C]: transpose each [Ct,Kt] tile block into [Kt,Ct].
        cb_reserve_back(cb_kdec_t, Kt * Ct);
        pack_reconfig_data_format(cb_kdec_t);
        reconfig_data_format_srca(cb_scr1);  // unary: in->srcA
        transpose_init(cb_scr1);
        for (uint32_t ki = 0; ki < Kt; ki++) {
            for (uint32_t ci = 0; ci < Ct; ci++) {
                tile_regs_acquire();
                transpose_tile(cb_scr1, ci * Kt + ki, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_kdec_t, ki * Ct + ci);
                tile_regs_release();
            }
        }
        cb_push_back(cb_kdec_t, Kt * Ct);
        WAIT(cb_kdec_t, Kt * Ct);
        POP(cb_scr1, ck);

        // ---- scan step (state read from cur_S) ----
        WAIT(cur_S, kv);                              // state from previous chunk
        mm(cb_w, cur_S, cb_scr1, Ct, Kt, Vt, false);  // v_prime = w @ S -> scr1
        WAIT(cb_scr1, cv);
        POP(cb_w, ck);
        ew(cb_u, cb_scr1, cb_vnew, cv, 1);  // v_new = u - v_prime
        WAIT(cb_vnew, cv);
        POP(cb_u, cv);
        POP(cb_scr1, cv);
        mm(cb_qdecay, cur_S, cb_ointer, Ct, Kt, Vt, false);  // o_inter = q_decay @ S
        WAIT(cb_ointer, cv);
        POP(cb_qdecay, ck);
        mm(cb_intra, cb_vnew, cb_scr1, Ct, Ct, Vt, false);  // intra_v = intra @ v_new
        WAIT(cb_scr1, cv);
        POP(cb_intra, cc);
        ew(cb_ointer, cb_scr1, cb_out, cv, 0);  // o = o_inter + intra_v
        POP(cb_ointer, cv);
        POP(cb_scr1, cv);
        mm(cb_kdec_t, cb_vnew, cb_supd, Kt, Ct, Vt, false);  // s_upd = k_dec_t @ v_new
        WAIT(cb_supd, kv);
        POP(cb_kdec_t, Kt * Ct);
        POP(cb_vnew, cv);
        // ---- S_new = cur_S*exp(decay_last) + s_upd, written to nxt_S (or cb_final on last).
        // Ping-pong avoids reading and writing the same state CB across the chunk boundary.
        const bool last = (c == NC - 1);
        // dl = exp(g_sum): for any row i, decayfac[i]*decay_exp[i]
        //   = exp(g_sum - decay_i)*exp(decay_i) = exp(g_sum). Tile 0's [0,0] element = dl,
        //   which is all bcast_scalar_mul reads. (scr2 is free during the scan.)
        ew(cb_decayfac, cb_decay_exp, cb_scr2, 1, 2);  // scr2[0] = dl (broadcast)
        WAIT(cb_scr2, 1);
        POP(cb_decayfac, Ct);
        POP(cb_decay_exp, Ct);
        bcast_scalar_mul(cur_S, cb_scr2, cb_stmp, kv);  // stmp = cur_S * dl
        WAIT(cb_stmp, kv);
        POP(cb_scr2, 1);
        POP(cur_S, kv);  // cur_S fully read -> drop it
        const uint32_t dst = last ? cb_final : nxt_S;
        ew(cb_stmp, cb_supd, dst, kv, 0);  // dst = stmp + s_upd
        POP(cb_stmp, kv);
        POP(cb_supd, kv);
    }
}
