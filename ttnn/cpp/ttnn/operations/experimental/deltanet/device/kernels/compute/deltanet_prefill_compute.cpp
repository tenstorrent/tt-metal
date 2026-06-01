// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for DeltaNet prefill (S>1 token loop).
// Identical recurrence to the S=1 full fused kernel, but loops S times
// with state ping-ponging between cb_state_A and cb_state_B.
// Initial state arrives in cb_state_init (reader-produced) to avoid
// producer conflicts — cb_state_A/B are exclusively compute-produced.
// Per-iteration CBs (norm_w, scaler, eps) are pushed/popped each iteration.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"

#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t cb_state_A    = get_compile_time_arg_val(0);
constexpr uint32_t cb_q          = get_compile_time_arg_val(1);
constexpr uint32_t cb_k          = get_compile_time_arg_val(2);
constexpr uint32_t cb_v          = get_compile_time_arg_val(3);
constexpr uint32_t cb_g          = get_compile_time_arg_val(4);
constexpr uint32_t cb_beta       = get_compile_time_arg_val(5);
constexpr uint32_t cb_output     = get_compile_time_arg_val(6);
constexpr uint32_t cb_state_B    = get_compile_time_arg_val(7);
constexpr uint32_t cb_tmp0       = get_compile_time_arg_val(8);
constexpr uint32_t cb_tmp1       = get_compile_time_arg_val(9);
constexpr uint32_t cb_acc        = get_compile_time_arg_val(10);
constexpr uint32_t Dk_tiles      = get_compile_time_arg_val(11);
constexpr uint32_t Dv_tiles      = get_compile_time_arg_val(12);
constexpr uint32_t cb_state_mid  = get_compile_time_arg_val(13);
constexpr uint32_t cb_k_T        = get_compile_time_arg_val(14);
constexpr uint32_t cb_z          = get_compile_time_arg_val(15);
constexpr uint32_t cb_norm_w     = get_compile_time_arg_val(16);
constexpr uint32_t cb_scaler     = get_compile_time_arg_val(17);
constexpr uint32_t cb_raw_out    = get_compile_time_arg_val(18);
constexpr uint32_t cb_eps        = get_compile_time_arg_val(19);
constexpr uint32_t S             = get_compile_time_arg_val(20);
constexpr uint32_t cb_state_init = get_compile_time_arg_val(21);

constexpr uint32_t state_tiles   = Dk_tiles * Dv_tiles;

void kernel_main() {
    binary_op_init_common(cb_state_A, cb_g, cb_state_mid);

    for (uint32_t s = 0; s < S; s++) {
        // Wait for per-iteration static CBs (reader pushes S copies)
        cb_wait_front(cb_norm_w, Dv_tiles);
        cb_wait_front(cb_scaler, 1);
        cb_wait_front(cb_eps, 1);

        // s=0: read from reader-produced cb_state_init, write to cb_state_A
        // s>0: ping-pong between A and B (both exclusively compute-produced)
        uint32_t cur_state = (s == 0) ? cb_state_init : ((s % 2 == 1) ? cb_state_A : cb_state_B);
        uint32_t new_state = (s % 2 == 0) ? cb_state_A : cb_state_B;

        // Step 1: S_mid = cur_state * decay (broadcast scalar multiply)
        {
            binary_op_init_common(cur_state, cb_g, cb_state_mid);
            mul_tiles_bcast_scalar_init_short(cur_state, cb_g);
            cb_wait_front(cur_state, state_tiles);
            cb_wait_front(cb_g, 1);
            cb_reserve_back(cb_state_mid, state_tiles);
            for (uint32_t t = 0; t < state_tiles; t++) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cur_state, cb_g, t, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_state_mid);
                tile_regs_release();
            }
            cb_push_back(cb_state_mid, state_tiles);
            cb_pop_front(cur_state, state_tiles);
        }

        // Step 2: mem = k @ S_mid → cb_tmp0 [Dv_tiles]
        {
            mm_init(cb_k, cb_state_mid, cb_tmp0);
            cb_wait_front(cb_state_mid, state_tiles);
            cb_wait_front(cb_k, Dk_tiles);
            cb_reserve_back(cb_tmp0, Dv_tiles);
            for (uint32_t j = 0; j < Dv_tiles; j++) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < Dk_tiles; i++) {
                    matmul_tiles(cb_k, cb_state_mid, i, i * Dv_tiles + j, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp0);
                tile_regs_release();
            }
            cb_push_back(cb_tmp0, Dv_tiles);
        }

        // Step 3: delta = (v - mem) * beta
        {
            binary_op_init_common(cb_v, cb_tmp0, cb_tmp1);
            sub_tiles_init(cb_v, cb_tmp0);
            cb_wait_front(cb_v, Dv_tiles);
            cb_wait_front(cb_tmp0, Dv_tiles);
            cb_reserve_back(cb_tmp1, Dv_tiles);
            for (uint32_t j = 0; j < Dv_tiles; j++) {
                tile_regs_acquire();
                sub_tiles(cb_v, cb_tmp0, j, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp1);
                tile_regs_release();
            }
            cb_push_back(cb_tmp1, Dv_tiles);
            cb_pop_front(cb_tmp0, Dv_tiles);

            binary_op_init_common(cb_tmp1, cb_beta, cb_acc);
            mul_tiles_bcast_scalar_init_short(cb_tmp1, cb_beta);
            cb_wait_front(cb_tmp1, Dv_tiles);
            cb_wait_front(cb_beta, 1);
            cb_reserve_back(cb_acc, Dv_tiles);
            for (uint32_t j = 0; j < Dv_tiles; j++) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_tmp1, cb_beta, j, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_acc);
                tile_regs_release();
            }
            cb_push_back(cb_acc, Dv_tiles);
            cb_pop_front(cb_tmp1, Dv_tiles);
        }

        // Step 4: S_new = S_mid + outer(k_T, delta)
        {
            cb_wait_front(cb_state_mid, state_tiles);
            cb_wait_front(cb_k_T, Dk_tiles);
            cb_wait_front(cb_acc, Dv_tiles);
            cb_reserve_back(new_state, state_tiles);

            for (uint32_t i = 0; i < Dk_tiles; i++) {
                for (uint32_t j = 0; j < Dv_tiles; j++) {
                    uint32_t state_tile_idx = i * Dv_tiles + j;

                    cb_reserve_back(cb_tmp1, 1);
                    mm_init(cb_k_T, cb_acc, cb_tmp1);
                    tile_regs_acquire();
                    matmul_tiles(cb_k_T, cb_acc, i, j, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_tmp1);
                    tile_regs_release();
                    cb_push_back(cb_tmp1, 1);

                    cb_wait_front(cb_tmp1, 1);
                    binary_op_init_common(cb_state_mid, cb_tmp1, new_state);
                    add_tiles_init(cb_state_mid, cb_tmp1);
                    tile_regs_acquire();
                    add_tiles(cb_state_mid, cb_tmp1, state_tile_idx, 0, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, new_state);
                    tile_regs_release();
                    cb_pop_front(cb_tmp1, 1);
                }
            }

            cb_push_back(new_state, state_tiles);
            cb_pop_front(cb_state_mid, state_tiles);
            cb_pop_front(cb_k_T, Dk_tiles);
            cb_pop_front(cb_acc, Dv_tiles);
        }

        // Step 5: raw_out = q @ S_new → cb_raw_out [Dv_tiles]
        // NOTE: Do NOT pop new_state — it's needed as next iteration's input
        {
            mm_init(cb_q, new_state, cb_raw_out);
            cb_wait_front(new_state, state_tiles);
            cb_wait_front(cb_q, Dk_tiles);
            cb_reserve_back(cb_raw_out, Dv_tiles);
            for (uint32_t j = 0; j < Dv_tiles; j++) {
                tile_regs_acquire();
                for (uint32_t i = 0; i < Dk_tiles; i++) {
                    matmul_tiles(cb_q, new_state, i, i * Dv_tiles + j, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_raw_out);
                tile_regs_release();
            }
            cb_push_back(cb_raw_out, Dv_tiles);
        }

        // Pop consumed per-token inputs
        cb_pop_front(cb_q, Dk_tiles);
        cb_pop_front(cb_k, Dk_tiles);
        cb_pop_front(cb_v, Dv_tiles);
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_beta, 1);

        // Steps 6-7: RMSNorm + silu gate → cb_output
        {
            cb_wait_front(cb_raw_out, Dv_tiles);
            cb_wait_front(cb_z, Dv_tiles);

            // 6a: output^2 → cb_tmp0
            binary_op_init_common(cb_raw_out, cb_raw_out, cb_tmp0);
            mul_tiles_init(cb_raw_out, cb_raw_out);
            cb_reserve_back(cb_tmp0, Dv_tiles);
            for (uint32_t t = 0; t < Dv_tiles; t++) {
                tile_regs_acquire();
                mul_tiles(cb_raw_out, cb_raw_out, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp0);
                tile_regs_release();
            }
            cb_push_back(cb_tmp0, Dv_tiles);

            // 6b: reduce_scalar(output^2) * scaler^2 → mean(x^2)
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp0, cb_scaler, cb_acc);
            cb_wait_front(cb_tmp0, Dv_tiles);
            cb_reserve_back(cb_acc, 1);
            tile_regs_acquire();
            for (uint32_t t = 0; t < Dv_tiles; t++) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp0, cb_scaler, t, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_acc);
            tile_regs_release();
            cb_push_back(cb_acc, 1);
            cb_pop_front(cb_tmp0, Dv_tiles);
            cb_pop_front(cb_scaler, 1);
            reduce_uninit(cb_acc);

            // 6c: rsqrt(mean(x^2) + eps)
            cb_wait_front(cb_acc, 1);
            cb_reserve_back(cb_tmp1, 1);
            binary_op_init_common(cb_acc, cb_eps, cb_tmp1);
            add_tiles_init(cb_acc, cb_eps);
            tile_regs_acquire();
            add_tiles(cb_acc, cb_eps, 0, 0, 0);
            rsqrt_tile_init();
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);
            cb_pop_front(cb_acc, 1);
            cb_pop_front(cb_eps, 1);

            // 6d: raw_out * rsqrt_val → cb_tmp0 (broadcast scalar)
            binary_op_init_common(cb_raw_out, cb_tmp1, cb_tmp0);
            mul_tiles_bcast_scalar_init_short(cb_raw_out, cb_tmp1);
            cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tmp0, Dv_tiles);
            for (uint32_t t = 0; t < Dv_tiles; t++) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_raw_out, cb_tmp1, t, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp0);
                tile_regs_release();
            }
            cb_push_back(cb_tmp0, Dv_tiles);
            cb_pop_front(cb_tmp1, 1);
            cb_pop_front(cb_raw_out, Dv_tiles);

            // 6e: normed * norm_weight → cb_acc
            binary_op_init_common(cb_tmp0, cb_norm_w, cb_acc);
            mul_tiles_init(cb_tmp0, cb_norm_w);
            cb_wait_front(cb_tmp0, Dv_tiles);
            cb_reserve_back(cb_acc, Dv_tiles);
            for (uint32_t t = 0; t < Dv_tiles; t++) {
                tile_regs_acquire();
                mul_tiles(cb_tmp0, cb_norm_w, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_acc);
                tile_regs_release();
            }
            cb_push_back(cb_acc, Dv_tiles);
            cb_pop_front(cb_tmp0, Dv_tiles);

            // 7a: silu(z) → cb_tmp0
            unary_op_init_common(cb_z, cb_tmp0);
            copy_tile_to_dst_init_short(cb_z);
            silu_tile_init();
            cb_reserve_back(cb_tmp0, Dv_tiles);
            for (uint32_t t = 0; t < Dv_tiles; t++) {
                tile_regs_acquire();
                copy_tile(cb_z, t, 0);
                silu_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp0);
                tile_regs_release();
            }
            cb_push_back(cb_tmp0, Dv_tiles);
            cb_pop_front(cb_z, Dv_tiles);

            // 7b: silu(z) * normed → cb_output
            binary_op_init_common(cb_tmp0, cb_acc, cb_output);
            mul_tiles_init(cb_tmp0, cb_acc);
            cb_wait_front(cb_tmp0, Dv_tiles);
            cb_wait_front(cb_acc, Dv_tiles);
            cb_reserve_back(cb_output, Dv_tiles);
            for (uint32_t t = 0; t < Dv_tiles; t++) {
                tile_regs_acquire();
                mul_tiles(cb_tmp0, cb_acc, t, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, Dv_tiles);
            cb_pop_front(cb_tmp0, Dv_tiles);
            cb_pop_front(cb_acc, Dv_tiles);
        }

        // Pop per-iteration norm_w so reader can push fresh copy
        // (scaler and eps are already popped inside the RMSNorm block above)
        cb_pop_front(cb_norm_w, Dv_tiles);
    }
}
