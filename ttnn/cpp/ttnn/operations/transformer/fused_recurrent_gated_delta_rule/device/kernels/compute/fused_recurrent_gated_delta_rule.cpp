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
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
constexpr uint32_t cb_out = 6, cb_state = 7, cb_s2 = 8, cb_s3 = 9, cb_sd = 10;
constexpr uint32_t cb_vread = 11, cb_u = 12, cb_kcol = 13;

// fp32 DST tiles available per tile_regs_acquire (half-sync).
constexpr uint32_t DST_TILES = 4;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must already be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
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

// S_new[Kt,Vt] = sd + kcol[Kt,1] (x) u[1,Vt]. Each DST tile is seeded with sd and the rank-1
// matmul accumulates onto it, so the outer product never round-trips through L1. When
// emit_state, the same DST tiles are also packed as the state output.
template <uint32_t Kt, uint32_t Vt>
void rank1_update(uint32_t sd, uint32_t kcol, uint32_t u, uint32_t o, bool emit_state) {
    constexpr uint32_t kv = Kt * Vt;
    cb_reserve_back(o, kv);
    if (emit_state) {
        cb_reserve_back(cb_state, kv);
    }
    pack_reconfig_data_format(o);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = (Vt - n0 < DST_TILES) ? (Vt - n0) : DST_TILES;
            tile_regs_acquire();
            reconfig_data_format_srca(sd);
            copy_init(sd);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(sd, mi * Vt + n0 + j, j);
            }
            reconfig_data_format(u, kcol);
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
    cb_push_back(o, kv);
    if (emit_state) {
        cb_push_back(cb_state, kv);
    }
}

// out = (A - B) * beta, n tiles, without spilling A - B to L1: FPU sub into DST[0], the beta
// tile into DST[1], then an SFPU multiply by DST[1]'s column 0 in place. beta sits at [0,0]
// and only row 0 of A/B carries data (rows 1..31 are zero), so row 0 is scaled by beta and
// the zero padding rows stay zero.
void delta_rule_residual(uint32_t a, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        reconfig_data_format(a, b);
        sub_tiles_init(a, b);
        sub_tiles(a, b, i, i, 0);
        reconfig_data_format_srca(beta);
        copy_init(beta);
        copy_tile(beta, 0, 1);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = A * scalar, n tiles. scalar is the [0,0] element of the single `scal` tile.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);
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

// out[Kt,1] = transpose of in[1,Kt]: transpose each of the Kt tiles. (in must be available.)
void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t per_token = get_compile_time_arg_val(2);
    const uint32_t T = get_arg_val<uint32_t>(0);

    constexpr uint32_t kv = Kt * Vt;

    compute_kernel_hw_startup(cb_q, cb_v, cb_out);

    for (uint32_t t = 0; t < T; t++) {
        const uint32_t cur_S = (t == 0) ? cb_S : ((t & 1u) ? cb_s2 : cb_s3);
        const uint32_t nxt_S = (t & 1u) ? cb_s3 : cb_s2;
        const bool last = (t == T - 1);

        // sd = cur_S * decay  (decay before read)
        WAIT(cb_decay, 1);
        WAIT(cur_S, kv);
        bcast_scalar_mul(cur_S, cb_decay, cb_sd, kv);
        POP(cb_decay, 1);
        POP(cur_S, kv);
        WAIT(cb_sd, kv);

        // vread = k . sd  ([1,V])
        WAIT(cb_k, Kt);
        mm(cb_k, cb_sd, cb_vread, 1, Kt, Vt, false);
        WAIT(cb_vread, Vt);

        // u = beta * (v - vread)
        WAIT(cb_v, Vt);
        WAIT(cb_beta, 1);
        delta_rule_residual(cb_v, cb_vread, cb_beta, cb_u, Vt);
        POP(cb_v, Vt);
        POP(cb_vread, Vt);
        POP(cb_beta, 1);
        WAIT(cb_u, Vt);

        // kcol = transpose(k) ([K,1])
        transpose_block(cb_k, cb_kcol, Kt);
        POP(cb_k, Kt);
        WAIT(cb_kcol, Kt);

        // S_new = sd + kcol (x) u -> nxt_S (and the state output, per token or last)
        rank1_update<Kt, Vt>(cb_sd, cb_kcol, cb_u, nxt_S, per_token || last);
        POP(cb_kcol, Kt);
        POP(cb_u, Vt);
        POP(cb_sd, kv);
        WAIT(nxt_S, kv);

        // o = q . S_new  (read from POST-update state)
        WAIT(cb_q, Kt);
        mm(cb_q, nxt_S, cb_out, 1, Kt, Vt, false);
        POP(cb_q, Kt);

        // nxt_S is intentionally NOT popped: the next iteration reads it as cur_S.
    }
}
