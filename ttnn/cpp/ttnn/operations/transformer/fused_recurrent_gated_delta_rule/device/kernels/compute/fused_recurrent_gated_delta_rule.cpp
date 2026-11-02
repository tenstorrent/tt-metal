// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel: the sequential-over-token recurrence for one head. Carries the recurrent state
// S [K,V] on-core across T tokens. Derived from flash-linear-attention
// `naive_recurrent_gated_delta_rule`; the vLLM `fused_sigmoid_gating_delta_rule_update` is this
// same recurrence over the K+1 speculative tokens.
//
// Per token (q pre-scaled + L2-normed, k L2-normed, decay = exp(g_t), all done host-side):
//   sd     = S * decay            (decay the state BEFORE the read, matches FLA + chunk scan)
//   vread  = k . sd               ([1,V])
//   u      = beta * (v - vread)   ([1,V])
//   S_new  = sd + k^T (x) u       ([K,V] rank-1 update)
//   o      = q . S_new            (read from the POST-update state)
//
// Per-token tiles are [1, D] (token in row 0, rows 1..31 host-zero-padded), so the outer-product
// update k^T (x) u is a matmul with inner dim 1 whose 31 padding lanes are zero and vanish.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
constexpr uint32_t cb_out = 6, cb_state = 7, cb_s2 = 8, cb_s3 = 9, cb_sd = 10;
constexpr uint32_t cb_vread = 11, cb_u = 12, cb_kcol = 13;
constexpr uint32_t cb_gdec = 14, cb_gbeta = 15;  // gb_fused: decay/beta computed here
constexpr uint32_t cb_l2ss = 18, cb_l2s = 19, cb_qn = 21, cb_kn = 22;  // fused_l2norm

// fp32 DST tiles available per tile_regs_acquire (half-sync).
constexpr uint32_t DST_TILES = 4;

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must already be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(Mt * Nt);
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
    cb_o.push_back(Mt * Nt);
}

// S_new[Kt,Vt] = sd + kcol[Kt,1] (x) u[1,Vt]. Each DST tile is seeded with sd and the rank-1
// matmul accumulates onto it, so the outer product never round-trips through L1. When
// emit_state, the same DST tiles are also packed as the state output.
template <uint32_t Kt, uint32_t Vt>
void rank1_update(uint32_t sd, uint32_t kcol, uint32_t u, uint32_t o, bool emit_state) {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_o(o);
    CircularBuffer cb_st(cb_state);
    cb_o.reserve_back(kv);
    if (emit_state) {
        cb_st.reserve_back(kv);
    }
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = (Vt - n0 < DST_TILES) ? (Vt - n0) : DST_TILES;
            tile_regs_acquire();
            copy_init(sd);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(sd, mi * Vt + n0 + j, j);
            }
            matmul_init(kcol, u, 0);
            for (uint32_t j = 0; j < nn; j++) {
                matmul_tiles(kcol, u, mi, n0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, o, mi * Vt + n0 + j);
                if (emit_state) {
                    pack_tile<true>(j, cb_state, mi * Vt + n0 + j);
                }
            }
            tile_regs_release();
        }
    }
    cb_o.push_back(kv);
    if (emit_state) {
        cb_st.push_back(kv);
    }
}

// out = (A - B) * beta, n tiles, without spilling A - B to L1. Per DST batch: FPU sub into
// DST[0..nn), the beta tile into DST[nn], then an SFPU multiply of each by DST[nn]'s column 0
// in place. beta sits at [0,0] and only row 0 of A/B carries data (rows 1..31 are zero), so
// row 0 is scaled by beta and the zero padding rows stay zero.
void delta_rule_residual(uint32_t a, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    constexpr uint32_t per_batch = DST_TILES - 1;  // the last DST slot holds beta
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t nn = (n - i0 < per_batch) ? (n - i0) : per_batch;
        tile_regs_acquire();
        sub_init(a, b);
        for (uint32_t j = 0; j < nn; j++) {
            sub_tiles(a, b, i0 + j, i0 + j, j);
        }
        copy_init(beta);
        copy_tile(beta, 0, nn);
        sfpu_mul_bcast_col_init();
        for (uint32_t j = 0; j < nn; j++) {
            sfpu_mul_bcast_col(j, nn);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// out = A * scalar, n tiles. scalar is the [0,0] element of the single `scal` tile.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// out[Kt,1] = transpose of in[1,Kt]: transpose each of the Kt tiles. (in must be available.)
void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

}  // namespace

// fused_l2norm: out = in * mult / sqrt(sum(in[0,:]^2) + 1e-6) over n tiles whose row 0 carries the vector (the
// sum of squares as in . in^T, whose [0,0] is the row-0 dot product; rows 1-31 are zero, so is the rest). eps is
// 1e-6, which is what rms_norm(x, epsilon=1e-6/K) * K^-0.5 amounts to.
void l2_normalize(uint32_t in, uint32_t out, uint32_t n, uint32_t mult_bits) {
    mm(in, in, cb_l2ss, 1, n, 1, true);
    CircularBuffer ss(cb_l2ss), sc(cb_l2s);
    ss.wait_front(1);
    sc.reserve_back(1);
    tile_regs_acquire();
    copy_init(cb_l2ss);
    copy_tile(cb_l2ss, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, 0x358637bdu);  // 1e-6f
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, mult_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_l2s);
    tile_regs_release();
    sc.push_back(1);
    ss.pop_front(1);
    sc.wait_front(1);
    bcast_scalar_mul(in, cb_l2s, out, n);
    sc.pop_front(1);
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t per_token = get_compile_time_arg_val(2);
    constexpr uint32_t gb_fused = get_compile_time_arg_val(3);
    constexpr uint32_t l2 = get_compile_time_arg_val(4);
    const uint32_t T = get_arg_val<uint32_t>(0);
    const uint32_t l2_scale_bits = get_arg_val<uint32_t>(1);
    // fused_l2norm reads the q/k it normalizes below; otherwise the host-normalized ones.
    constexpr uint32_t q_id = l2 ? cb_qn : cb_q;
    constexpr uint32_t k_id = l2 ? cb_kn : cb_k;
    // gb_fused reads the decay/beta it computes below; otherwise the host-computed ones.
    constexpr uint32_t dec_id = gb_fused ? cb_gdec : cb_decay;
    constexpr uint32_t bet_id = gb_fused ? cb_gbeta : cb_beta;

    constexpr uint32_t kv = Kt * Vt;

    // Every CB is fp32 (see the program factory), so the unpack/pack formats configured here hold
    // for every op below; no per-op data-format reconfig is needed.
    compute_kernel_hw_startup(cb_q, cb_v, cb_out);

    CircularBuffer q(q_id), k(k_id), v(cb_v), decay(dec_id), beta(bet_id);
    CircularBuffer sd(cb_sd), vread(cb_vread), u(cb_u), kcol(cb_kcol);

    if constexpr (gb_fused) {
        // decay = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b): elementwise on four zero tiles that
        // carry the value at [0,0] (a, dt_bias, -exp(A_log) in cb_decay; b in cb_beta). fp32 throughout.
        CircularBuffer raw(cb_decay), rawb(cb_beta), dec(cb_gdec), bet(cb_gbeta);
        raw.wait_front(3);
        rawb.wait_front(1);
        dec.reserve_back(1);
        tile_regs_acquire();
        copy_init(cb_decay);
        copy_tile(cb_decay, 0, 0);
        copy_tile(cb_decay, 1, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        softplus_tile_init();
        softplus_tile(0, 0x3f800000u, 0x3f800000u, 0x41a00000u);  // beta 1, 1/beta 1, threshold 20 (= torch)
        copy_init(cb_decay);
        copy_tile(cb_decay, 2, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gdec);
        tile_regs_release();
        dec.push_back(1);
        bet.reserve_back(1);
        tile_regs_acquire();
        copy_init(cb_beta);
        copy_tile(cb_beta, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gbeta);
        tile_regs_release();
        bet.push_back(1);
        raw.pop_front(3);
        rawb.pop_front(1);
    }

    for (uint32_t t = 0; t < T; t++) {
        const uint32_t cur_S_id = (t == 0) ? cb_S : ((t & 1u) ? cb_s2 : cb_s3);
        const uint32_t nxt_S_id = (t & 1u) ? cb_s3 : cb_s2;
        CircularBuffer cur_S(cur_S_id), nxt_S(nxt_S_id);
        const bool last = (t == T - 1);

        if constexpr (l2) {
            CircularBuffer kraw(cb_k), qraw(cb_q);
            kraw.wait_front(Kt);
            l2_normalize(cb_k, cb_kn, Kt, 0x3f800000u);  // k / ||k||
            kraw.pop_front(Kt);
            qraw.wait_front(Kt);
            l2_normalize(cb_q, cb_qn, Kt, l2_scale_bits);  // q * scale / ||q||
            qraw.pop_front(Kt);
        }

        // sd = cur_S * decay  (decay before read)
        decay.wait_front(1);
        cur_S.wait_front(kv);
        bcast_scalar_mul(cur_S_id, dec_id, cb_sd, kv);
        decay.pop_front(1);
        cur_S.pop_front(kv);
        sd.wait_front(kv);

        // vread = k . sd  ([1,V])
        k.wait_front(Kt);
        mm(k_id, cb_sd, cb_vread, 1, Kt, Vt, false);
        vread.wait_front(Vt);

        // u = beta * (v - vread)
        v.wait_front(Vt);
        beta.wait_front(1);
        delta_rule_residual(cb_v, cb_vread, bet_id, cb_u, Vt);
        v.pop_front(Vt);
        vread.pop_front(Vt);
        beta.pop_front(1);
        u.wait_front(Vt);

        // kcol = transpose(k) ([K,1])
        transpose_block(k_id, cb_kcol, Kt);
        k.pop_front(Kt);
        kcol.wait_front(Kt);

        // S_new = sd + kcol (x) u -> nxt_S (and the state output, per token or last)
        rank1_update<Kt, Vt>(cb_sd, cb_kcol, cb_u, nxt_S_id, per_token || last);
        kcol.pop_front(Kt);
        u.pop_front(Vt);
        sd.pop_front(kv);
        nxt_S.wait_front(kv);

        // o = q . S_new  (read from POST-update state)
        q.wait_front(Kt);
        mm(q_id, nxt_S_id, cb_out, 1, Kt, Vt, false);
        q.pop_front(Kt);

        // nxt_S is intentionally NOT popped: the next iteration reads it as cur_S.
    }
}
