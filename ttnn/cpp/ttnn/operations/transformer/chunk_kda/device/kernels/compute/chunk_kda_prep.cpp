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
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
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
// PHASE A output for the scan step's state decay (reuses a scan-only index, unused in prep).
constexpr uint32_t cb_dl = cb_vnew;
// WY-inverse quadrant masks (3 tiles: 0=Qtl, 1=Qbr, 2=Q10). Reuses the cb_u slot (unused in
// the stable-form prep); the reader loads them once. Used only by invert_block.
constexpr uint32_t cb_mask = cb_u;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

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

// KDA: o = exp(-in), n tiles. (negate the dst reg with a scalar mul, then exp.)
void expc_neg(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    binop_with_scalar_tile_init();
    exp_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        mul_unary_tile(0, 0xBF800000u);  // * -1.0f
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

// out[0] = copy of src[src_tile] (single 32x32 tile). src must be available.
void cpy_t(uint32_t src, uint32_t src_tile, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(src);
    copy_tile_to_dst_init_short(src);
    tile_regs_acquire();
    copy_tile(src, src_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// out[0] = a[ai] (op) b[bi], single tile. op: 0 add, 2 mul. (Like ew but with free tile indices.)
void ewt(uint32_t a, uint32_t ai, uint32_t b, uint32_t bi, uint32_t o, int op) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    if (op == 0) {
        add_tiles_init(a, b);
    } else {
        mul_tiles_init(a, b);
    }
    tile_regs_acquire();
    if (op == 0) {
        add_tiles(a, b, ai, bi, 0);
    } else {
        mul_tiles(a, b, ai, bi, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// (I32 - Nq)^-1 for a strictly-lower 16-block Nq isolated in one 16-quadrant (rest zero),
// nilpotent at 16. Horner in 15 terms -> out (single tile); the other diagonal quadrant is I.
// Small block + short chain keeps fp32 bounded where a 32x32/31-term Horner cancels.
void invert16(uint32_t nq, uint32_t out, uint32_t tmp) {
    ew(cb_eye, nq, out, 1, 0);  // out = I + Nq
    CircularBuffer(out).wait_front(1);
    for (uint32_t m = 2; m < 16; m++) {  // sum_{k<16} Nq^k
        mm(nq, out, tmp, 1, 1, 1, false);
        CircularBuffer(tmp).wait_front(1);
        CircularBuffer(out).pop_front(1);
        ew(cb_eye, tmp, out, 1, 0);  // out = I + Nq @ out
        CircularBuffer(out).wait_front(1);
        CircularBuffer(tmp).pop_front(1);
    }
}

// Assemble the 2x2 tile-block matrix [[s0[t0], s1[t1]], [s2[t2], s3[t3]]] into o (4 tiles).
void asm4(
    uint32_t s0,
    uint32_t t0,
    uint32_t s1,
    uint32_t t1,
    uint32_t s2,
    uint32_t t2,
    uint32_t s3,
    uint32_t t3,
    uint32_t o) {
    const uint32_t src[4] = {s0, s1, s2, s3};
    const uint32_t tl[4] = {t0, t1, t2, t3};
    cb_reserve_back(o, 4);
    pack_reconfig_data_format(o);
    for (uint32_t i = 0; i < 4; i++) {
        reconfig_data_format_srca(src[i]);
        copy_tile_to_dst_init_short(src[i]);
        tile_regs_acquire();
        copy_tile(src[i], tl[i], 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, 4);
}

// Invert one 32x32 diagonal tile-block: out[0] = (I32 - negN)^-1, negN = src[tile] (strictly-lower
// 32x32). Mirrors FLA solve_tril: split into 16-quadrants negN = [[N00,0],[N10,N11]], invert the two
// diagonal 16-blocks (short, bounded Horners), and form the off-diagonal EXACTLY (one matmul chain,
// no power series). A single 32x32 Horner instead loses fp32 precision on harder blocks.
//   Bi00=(I-N00)^-1 (top-left), Bi11=(I-N11)^-1 (bottom-right), off=Bi11@N10@Bi00 (bottom-left).
//   out = [[Bi00,0],[off,Bi11]].
// Masks (single tiles): cb_mask[0]=Qtl, [1]=Qbr, [2]=Q10 (bottom-left). tmpN/tmpT = scratch.
// Private scratch A..D use cb_ointer/cb_final/cb_s2/cb_s3 — all fp32 and NOT drained by the prep
// writer (unlike the output CBs cb_w/cb_qdecay/cb_intra, whose scratch pushes the writer would
// wrongly consume). None alias src (cb_scr3), out, or the Ct==2 persistents (cb_supd/cb_stmp).
void invert_block(uint32_t src, uint32_t tile, uint32_t out, uint32_t tmpN, uint32_t tmpT) {
    const uint32_t A = cb_S, B = cb_final, C = cb_s2, D = cb_s3;
    cpy_t(src, tile, tmpN);
    CircularBuffer(tmpN).wait_front(1);  // negN -> tmpN[0]
    // Bi00 = (I-N00)^-1  (N00 = top-left quadrant of negN; top-right is already 0)
    ewt(tmpN, 0, cb_mask, 0, A, 2);
    CircularBuffer(A).wait_front(1);  // N00
    invert16(A, B, tmpT);
    CircularBuffer(B).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi00 -> B
    // Bi11 = (I-N11)^-1  (N11 = bottom-right quadrant)
    ewt(tmpN, 0, cb_mask, 1, A, 2);
    CircularBuffer(A).wait_front(1);  // N11
    invert16(A, C, tmpT);
    CircularBuffer(C).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi11 -> C
    // off = Bi11 @ N10 @ Bi00  (N10 = bottom-left quadrant; result lives only there)
    ewt(tmpN, 0, cb_mask, 2, A, 2);
    CircularBuffer(A).wait_front(1);  // N10
    CircularBuffer(tmpN).pop_front(1);
    mm(C, A, tmpT, 1, 1, 1, false);
    CircularBuffer(tmpT).wait_front(1);
    CircularBuffer(A).pop_front(1);  // Bi11@N10
    mm(tmpT, B, A, 1, 1, 1, false);
    CircularBuffer(A).wait_front(1);
    CircularBuffer(tmpT).pop_front(1);  // @Bi00 -> A(off)
    // out = Qtl*Bi00 + Qbr*Bi11 + off
    ewt(B, 0, cb_mask, 0, D, 2);
    CircularBuffer(D).wait_front(1);
    CircularBuffer(B).pop_front(1);  // Bi00_tl -> D
    ewt(C, 0, cb_mask, 1, B, 2);
    CircularBuffer(B).wait_front(1);
    CircularBuffer(C).pop_front(1);  // Bi11_br -> B
    ewt(D, 0, B, 0, C, 0);
    CircularBuffer(C).wait_front(1);
    CircularBuffer(D).pop_front(1);
    CircularBuffer(B).pop_front(1);
    ewt(C, 0, A, 0, out, 0);
    CircularBuffer(C).pop_front(1);
    CircularBuffer(A).pop_front(1);  // + off -> out
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

// OPT-A/B in-kernel L2-norm over K. rowsum_k: o[Mt,1(broadcast)] = sum over the full K dim of
// in[Mt,Kt], computed as in @ ones by reusing cb_ones tile 0 as the [K,1] contraction operand
// (avoids a dedicated ones-column constant). Mirrors the `mm` helper's reconfig/matmul discipline.
void rowsum_k(uint32_t in, uint32_t o, uint32_t Mt, uint32_t Kt) {
    cb_reserve_back(o, Mt);
    pack_reconfig_data_format(o);
    reconfig_data_format(cb_ones, in);  // matmul(in, cb_ones): in->srcB, cb_ones->srcA
    matmul_init(in, cb_ones, 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < Kt; ki++) {
            matmul_tiles(in, cb_ones, mi * Kt + ki, 0, 0);  // reuse ones tile 0 for every ki
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, mi);
        tile_regs_release();
    }
    cb_push_back(o, Mt);
}

// inv_rms: o[i] = rsqrt(in[i] + eps) [* scale]. in holds per-row sum-of-squares (rowsum_k output);
// out is the per-row inverse-L2 factor (optionally pre-scaled, for folding q's scale into the norm).
// eps/scale arrive as fp32-bit-cast uint32 compile args.
void inv_rms(uint32_t in, uint32_t o, uint32_t n, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);  // + eps
        rsqrt_tile_init();
        rsqrt_tile(0);  // 1/sqrt(sumsq + eps)
        if (do_scale) {
            binop_with_scalar_tile_init();
            mul_unary_tile(0, scale_bits);  // * scale (q only)
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    // OPT-B: QK_NORM=1 => L2-normalize q/k over K in-kernel (host skipped it), folding q's `scale`
    // into the norm. scale/eps arrive as fp32 bits. Only valid for Ct==1 (uses cb_supd/cb_stmp, which
    // are free outside the Ct==2 inverse branch) — the op host gates QK_NORM on chunk_size==32.
    constexpr uint32_t QK_NORM = get_compile_time_arg_val(3);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(5);
    // Chunk-parallel: NC here is this core's local work-item count (chunks assigned to it), NOT the
    // sequence-wide chunk count. Each work-item is an independent (head, chunk) prep — no cross-item
    // state — so the loop just processes `NC` items regardless of which (h, c) they map to.
    const uint32_t NC = get_arg_val<uint32_t>(0);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t C = Ct * 32;

    compute_kernel_hw_startup(cb_q, cb_k, cb_u);

    // Constants (loaded once by reader). Initial state is in cb_S (reader pushed it).
    WAIT(cb_eye, cc);
    WAIT(cb_tril, cc);
    WAIT(cb_ones, cc);
    WAIT(cb_mask, 3);  // Qtl, Qbr, Q10 (used by invert_block)

    // PHASE A (prep): state-independent per-chunk quantities. No recurrent state here; the
    // sequential state scan lives in the separate scan kernel. Outputs (per chunk) u, w, k_dec_t,
    // q_decay, intra, dl are pushed to their CBs and streamed to DRAM by the prep writer.
    for (uint32_t c = 0; c < NC; c++) {
        WAIT(cb_q, ck);
        WAIT(cb_k, ck);
        WAIT(cb_v, cv);
        WAIT(cb_g, Ct);
        WAIT(cb_beta, Ct);

        // ---- OPT-B: in-kernel L2-norm of q,k over K (fold q's scale). Consumes the raw reader q/k
        // and produces normalized q->cb_supd, k->cb_stmp (both free in Ct==1). The rest of the chunk
        // then reads Q/Kk instead of cb_q/cb_k. scr1/scr2/scr3 are free here (used only later). ----
        uint32_t Q = cb_q, Kk = cb_k;
        if constexpr (QK_NORM) {
            // q: q^2 -> rowsum_K -> rsqrt(+eps)*scale -> q_normed (cb_supd)
            ew(cb_q, cb_q, cb_scr1, ck, 2);
            WAIT(cb_scr1, ck);
            rowsum_k(cb_scr1, cb_scr2, Ct, Kt);
            WAIT(cb_scr2, Ct);
            POP(cb_scr1, ck);
            inv_rms(cb_scr2, cb_scr3, Ct, EPS_BITS, SCALE_BITS, /*do_scale=*/true);
            WAIT(cb_scr3, Ct);
            POP(cb_scr2, Ct);
            bcast_cols_mul(cb_q, cb_scr3, cb_supd, Ct, Kt);
            WAIT(cb_supd, ck);
            POP(cb_scr3, Ct);
            POP(cb_q, ck);
            // k: same, no scale -> k_normed (cb_stmp)
            ew(cb_k, cb_k, cb_scr1, ck, 2);
            WAIT(cb_scr1, ck);
            rowsum_k(cb_scr1, cb_scr2, Ct, Kt);
            WAIT(cb_scr2, Ct);
            POP(cb_scr1, ck);
            inv_rms(cb_scr2, cb_scr3, Ct, EPS_BITS, SCALE_BITS, /*do_scale=*/false);
            WAIT(cb_scr3, Ct);
            POP(cb_scr2, Ct);
            bcast_cols_mul(cb_k, cb_scr3, cb_stmp, Ct, Kt);
            WAIT(cb_stmp, ck);
            POP(cb_scr3, Ct);
            POP(cb_k, ck);
            Q = cb_supd;
            Kk = cb_stmp;
        }

        // ---- P1: v_beta, k_beta ----
        bcast_cols_mul(cb_v, cb_beta, cb_vbeta, Ct, Vt);
        WAIT(cb_vbeta, cv);
        bcast_cols_mul(Kk, cb_beta, cb_kbeta, Ct, Kt);
        WAIT(cb_kbeta, ck);
        POP(cb_beta, Ct);
        POP(cb_v, cv);

        // ---- KDA P2: decay=cumsum(g) [C,K]; eg=exp(decay); k_keng=k(.)exp(-decay); decayfac; w ----
        // Per-channel (diagonal) gate: the scalar [C,1] decay / [C,C] L_mask become [C,K]. The L_mask is
        // deleted — decay is absorbed into the matmul operands (A=(k(.)eg)@(k(.)eng)^T). cb_lmask is
        // repurposed to hold k_keng=k(.)exp(-decay); w=k_beta(.)eg is hoisted here (also A's operand).
        mm(cb_tril, cb_g, cb_decay, Ct, Ct, Kt, false);   // decay = tril @ g   [C,K]
        WAIT(cb_decay, ck);
        expc(cb_decay, cb_decay_exp, ck);                 // eg = exp(decay)    [C,K]
        WAIT(cb_decay_exp, ck);
        expc_neg(cb_decay, cb_scr1, ck);                  // eng = exp(-decay)  [C,K]
        WAIT(cb_scr1, ck);
        ew(Kk, cb_scr1, cb_lmask, ck, 2);                 // k_keng = k (.) eng [C,K] (persists past inverse)
        WAIT(cb_lmask, ck);
        POP(cb_scr1, ck);
        // decayfac = exp(g_sum - decay); g_sum = ones @ g  [C,K]
        mm(cb_ones, cb_g, cb_scr1, Ct, Ct, Kt, false);    // g_sum broadcast to every row [C,K]
        WAIT(cb_scr1, ck);
        POP(cb_g, ck);
        ew(cb_scr1, cb_decay, cb_scr2, ck, 1);            // g_sum - decay
        WAIT(cb_scr2, ck);
        POP(cb_scr1, ck);
        POP(cb_decay, ck);
        expc(cb_scr2, cb_decayfac, ck);                   // decayfac [C,K]
        WAIT(cb_decayfac, ck);
        POP(cb_scr2, ck);
        // w = k_beta (.) eg  (WY state operand AND factored operand for A)
        ew(cb_kbeta, cb_decay_exp, cb_w, ck, 2);          // w = k_beta * eg [C,K]
        WAIT(cb_w, ck);
        POP(cb_kbeta, ck);

        // ---- N = strictly_lower(k_beta@k^T * L_mask); T_inv = (I + strictly_lower)^-1 ----
        // The WY inverse, mirroring FLA's solve_tril: block down to 16x16 (invert_block splits each
        // 32x32 tile into 16-quadrants), invert the small diagonal blocks with bounded Horners, and
        // merge off-diagonal blocks EXACTLY. This keeps every intermediate bounded, unlike a single
        // 32x32/full-matrix Horner whose deep power series loses fp32 precision on harder chunks.
        mm(cb_w, cb_lmask, cb_scr1, Ct, Kt, Ct, true);  // A_full = w @ k_keng^T [C,C] (cb_lmask = k_keng)
        WAIT(cb_scr1, cc);
        ew(cb_scr1, cb_tril, cb_scr2, cc, 2);  // A_masked = A_full * tril (lower incl) -> cb_scr2
        WAIT(cb_scr2, cc);
        POP(cb_scr1, cc);
        ew(cb_scr2, cb_eye, cb_scr1, cc, 2);  // diag(kk_masked)
        WAIT(cb_scr1, cc);
        // negN = diag - kk_masked = -(strictly_lower(kk_masked))  (= -A_strict, kept in cb_scr3)
        ew(cb_scr1, cb_scr2, cb_scr3, cc, 1);
        WAIT(cb_scr3, cc);
        POP(cb_scr1, cc);
        POP(cb_scr2, cc);

        if constexpr (Ct == 1) {
            // Single 32x32 block: T_inv is just its inverse.
            invert_block(cb_scr3, 0, cb_Tinv, cb_scr1, cb_scr2);
            WAIT(cb_Tinv, cc);
            POP(cb_scr3, cc);
        } else if constexpr (Ct == 2) {
            // 2x2 tile-block lower-triangular. negN tiles: 0=(0,0), 2=(1,0), 3=(1,1); (0,1)=0.
            // Diagonal inverses Mi11, Mi22, then off-diagonal Mi21 = -Mi22 @ A21 @ Mi11.
            // (A21 = -negN21, so -Mi22@A21@Mi11 = Mi22 @ negN21 @ Mi11.)
            // Mi11 -> cb_supd, Mi22 -> cb_stmp, Mi21 -> cb_ointer (all free in prep).
            invert_block(cb_scr3, 0, cb_supd, cb_scr1, cb_scr2);  // Mi11
            invert_block(cb_scr3, 3, cb_stmp, cb_scr1, cb_scr2);  // Mi22
            cpy_t(cb_scr3, 2, cb_scr1);                           // negN21 -> cb_scr1[0]
            WAIT(cb_scr1, 1);
            mm(cb_scr1, cb_supd, cb_scr2, 1, 1, 1, false);  // tmp = negN21 @ Mi11
            WAIT(cb_scr2, 1);
            POP(cb_scr1, 1);
            mm(cb_stmp, cb_scr2, cb_ointer, 1, 1, 1, false);  // Mi21 = Mi22 @ tmp
            WAIT(cb_ointer, 1);
            POP(cb_scr2, 1);
            POP(cb_scr3, cc);  // negN done
            // T_inv = [[Mi11, 0], [Mi21, Mi22]]  (cb_eye[1] is the zero block)
            asm4(cb_supd, 0, cb_eye, 1, cb_ointer, 0, cb_stmp, 0, cb_Tinv);
            WAIT(cb_Tinv, cc);
            POP(cb_supd, 1);
            POP(cb_stmp, 1);
            POP(cb_ointer, 1);
        } else {
            // Fallback (C>64, currently xfail): full-matrix Horner.
            ew(cb_eye, cb_scr3, cb_Tinv, cc, 0);
            WAIT(cb_Tinv, cc);
            for (uint32_t m = 2; m < C; m++) {
                mm(cb_scr3, cb_Tinv, cb_scr1, Ct, Ct, Ct, false);
                WAIT(cb_scr1, cc);
                POP(cb_Tinv, cc);
                ew(cb_eye, cb_scr1, cb_Tinv, cc, 0);
                WAIT(cb_Tinv, cc);
                POP(cb_scr1, cc);
            }
            POP(cb_scr3, cc);
        }

        // cb_vbeta (v_beta), cb_w (kd=k_beta*eg, from P2), cb_Tinv (T_inv) remain pushed for the writer.

        // ---- q_decay = q (.) eg ; intra = q_decay @ k_keng^T (lower-incl) ----
        ew(Q, cb_decay_exp, cb_qdecay, ck, 2);            // q_decay = q * eg [C,K]
        WAIT(cb_qdecay, ck);
        POP(Q, ck);
        mm(cb_qdecay, cb_lmask, cb_scr1, Ct, Kt, Ct, true);  // qk_full = q_decay @ k_keng^T [C,C]
        WAIT(cb_scr1, cc);
        ew(cb_scr1, cb_tril, cb_intra, cc, 2);            // intra = qk_full * tril (lower incl)
        WAIT(cb_intra, cc);
        POP(cb_scr1, cc);
        POP(cb_lmask, ck);  // k_keng done

        // ---- k_dec = k (.) decayfac ; k_dec_t = transpose(k_dec) [K,C] ----
        ew(Kk, cb_decayfac, cb_scr1, ck, 2);              // k_dec = k * decayfac [C,K]
        WAIT(cb_scr1, ck);
        POP(Kk, ck);
        cb_reserve_back(cb_kdec_t, Kt * Ct);
        pack_reconfig_data_format(cb_kdec_t);
        reconfig_data_format_srca(cb_scr1);
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
        POP(cb_scr1, ck);

        // ---- dl = exp(g_sum) per-K, as [K,1]: dl_bcast = decayfac (.) eg [C,K] (all rows equal =
        // exp(g_sum)); transpose each of the Kt tiles (row->col) so the scan's bcast_cols_mul reads col 0.
        ew(cb_decayfac, cb_decay_exp, cb_scr1, ck, 2);    // dl_bcast [C,K]
        WAIT(cb_scr1, ck);
        POP(cb_decayfac, ck);
        POP(cb_decay_exp, ck);
        cb_reserve_back(cb_dl, Kt);
        pack_reconfig_data_format(cb_dl);
        reconfig_data_format_srca(cb_scr1);
        transpose_init(cb_scr1);
        for (uint32_t ki = 0; ki < Kt; ki++) {
            tile_regs_acquire();
            transpose_tile(cb_scr1, ki, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dl, ki);
            tile_regs_release();
        }
        cb_push_back(cb_dl, Kt);
        POP(cb_scr1, ck);
        // u, w, k_dec_t, q_decay, intra, dl remain pushed in their CBs -> prep writer -> DRAM.
        // (They are NOT popped here; the writer drains them per chunk.)
    }
}
