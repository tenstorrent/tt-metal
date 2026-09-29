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
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
constexpr uint32_t cb_out = 6, cb_state = 7, cb_s2 = 8, cb_s3 = 9, cb_sd = 10;
constexpr uint32_t cb_vread = 11, cb_u = 12, cb_kcol = 13, cb_supd = 14, cb_delta = 15;

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

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
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

// out = copy of in (front, no pop), n tiles.
void cp(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
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

        // delta = v - vread ; u = beta * delta
        WAIT(cb_v, Vt);
        ew(cb_v, cb_vread, cb_delta, Vt, 1);
        POP(cb_v, Vt);
        POP(cb_vread, Vt);
        WAIT(cb_delta, Vt);
        WAIT(cb_beta, 1);
        bcast_scalar_mul(cb_delta, cb_beta, cb_u, Vt);
        POP(cb_beta, 1);
        POP(cb_delta, Vt);
        WAIT(cb_u, Vt);

        // kcol = transpose(k) ([K,1]); supd = kcol (x) u  ([K,V] rank-1 outer)
        transpose_block(cb_k, cb_kcol, Kt);
        POP(cb_k, Kt);
        WAIT(cb_kcol, Kt);
        mm(cb_kcol, cb_u, cb_supd, Kt, 1, Vt, false);
        POP(cb_kcol, Kt);
        POP(cb_u, Vt);
        WAIT(cb_supd, kv);

        // S_new = sd + supd -> nxt_S
        ew(cb_sd, cb_supd, nxt_S, kv, 0);
        POP(cb_sd, kv);
        POP(cb_supd, kv);
        WAIT(nxt_S, kv);

        // o = q . S_new  (read from POST-update state)
        WAIT(cb_q, Kt);
        mm(cb_q, nxt_S, cb_out, 1, Kt, Vt, false);
        POP(cb_q, Kt);

        // state output: per token, or just the final state
        if (per_token || last) {
            cp(nxt_S, cb_state, kv);
        }
        // nxt_S is intentionally NOT popped: the next iteration reads it as cur_S.
    }
}
