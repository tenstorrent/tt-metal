// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for fully fused DeltaNet decode.
// All CBs are bf16 (uniform format). State precision maintained via
// f32 DRAM storage with reader/writer format conversion.
// fp32_dest_acc_en=true gives f32 precision for intermediate DST operations.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t cb_state_in   = get_compile_time_arg_val(0);
constexpr uint32_t cb_q          = get_compile_time_arg_val(1);
constexpr uint32_t cb_k          = get_compile_time_arg_val(2);
constexpr uint32_t cb_v          = get_compile_time_arg_val(3);
constexpr uint32_t cb_g          = get_compile_time_arg_val(4);
constexpr uint32_t cb_beta       = get_compile_time_arg_val(5);
constexpr uint32_t cb_output     = get_compile_time_arg_val(6);
constexpr uint32_t cb_state_out  = get_compile_time_arg_val(7);
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

constexpr uint32_t state_tiles   = Dk_tiles * Dv_tiles;

void kernel_main() {
    // All CBs are bf16 — single init is sufficient.
    binary_op_init_common(cb_state_in, cb_g, cb_state_mid);

    // Step 1: S_mid = S * decay (broadcast scalar multiply)
    {
        mul_tiles_bcast_scalar_init_short(cb_state_in, cb_g);
        cb_wait_front(cb_state_in, state_tiles);
        cb_wait_front(cb_g, 1);
        cb_reserve_back(cb_state_mid, state_tiles);
        for (uint32_t t = 0; t < state_tiles; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_state_in, cb_g, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_mid);
            tile_regs_release();
        }
        cb_push_back(cb_state_mid, state_tiles);
        cb_pop_front(cb_state_in, state_tiles);
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
        // v - mem → cb_tmp1
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

        // (v - mem) * beta → cb_acc
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
        cb_reserve_back(cb_state_out, state_tiles);

        for (uint32_t i = 0; i < Dk_tiles; i++) {
            for (uint32_t j = 0; j < Dv_tiles; j++) {
                uint32_t state_tile_idx = i * Dv_tiles + j;

                // outer product: k_T @ delta → cb_tmp1 (one tile)
                cb_reserve_back(cb_tmp1, 1);
                mm_init(cb_k_T, cb_acc, cb_tmp1);
                tile_regs_acquire();
                matmul_tiles(cb_k_T, cb_acc, i, j, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_tmp1);
                tile_regs_release();
                cb_push_back(cb_tmp1, 1);

                // S_mid + outer_tile → S_out
                cb_wait_front(cb_tmp1, 1);
                binary_op_init_common(cb_state_mid, cb_tmp1, cb_state_out);
                add_tiles_init(cb_state_mid, cb_tmp1);
                tile_regs_acquire();
                add_tiles(cb_state_mid, cb_tmp1, state_tile_idx, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_state_out);
                tile_regs_release();
                cb_pop_front(cb_tmp1, 1);
            }
        }

        cb_push_back(cb_state_out, state_tiles);
        cb_pop_front(cb_state_mid, state_tiles);
        cb_pop_front(cb_k_T, Dk_tiles);
        cb_pop_front(cb_acc, Dv_tiles);
    }

    // Step 5: raw_out = q @ S_new → cb_raw_out [Dv_tiles]
    {
        mm_init(cb_q, cb_state_out, cb_raw_out);
        cb_wait_front(cb_state_out, state_tiles);
        cb_wait_front(cb_q, Dk_tiles);
        cb_reserve_back(cb_raw_out, Dv_tiles);
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < Dk_tiles; i++) {
                matmul_tiles(cb_q, cb_state_out, i, i * Dv_tiles + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_raw_out);
            tile_regs_release();
        }
        cb_push_back(cb_raw_out, Dv_tiles);
    }

    // Pop consumed input CBs
    cb_pop_front(cb_q, Dk_tiles);
    cb_pop_front(cb_k, Dk_tiles);
    cb_pop_front(cb_v, Dv_tiles);
    cb_pop_front(cb_g, 1);
    cb_pop_front(cb_beta, 1);

    // Output the RAW pre-norm read (q @ S_new). This op's contract for the Qwen3.6 GDN port is
    // raw-o (matching recurrent_gated_delta_rule_decode_ttnn): the caller does the gated-RMSNorm
    // + silu(z) in ttnn. Folding norm/gate into this kernel is a NET LOSS on Blackhole — the op is
    // one-core-per-head, so the RMSNorm reduce/rsqrt serializes per head (+8.4ms/step at B=8),
    // whereas the ttnn tail runs the elementwise norm/gate parallel across many cores. z_proj /
    // norm_weight / a_log / dt_bias are accepted but unused here; we consume their CBs to keep the
    // reader/compute pipeline balanced.
    {
        cb_wait_front(cb_raw_out, Dv_tiles);
        cb_reserve_back(cb_output, Dv_tiles);
        copy_tile_to_dst_init_short(cb_raw_out);
        for (uint32_t t = 0; t < Dv_tiles; t++) {
            tile_regs_acquire();
            copy_tile(cb_raw_out, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);
            tile_regs_release();
        }
        cb_push_back(cb_output, Dv_tiles);
        cb_pop_front(cb_raw_out, Dv_tiles);
        cb_wait_front(cb_z, Dv_tiles);
        cb_pop_front(cb_z, Dv_tiles);
        cb_wait_front(cb_norm_w, Dv_tiles);
        cb_pop_front(cb_norm_w, Dv_tiles);
        cb_wait_front(cb_scaler, 1);
        cb_pop_front(cb_scaler, 1);
        cb_wait_front(cb_eps, 1);
        cb_pop_front(cb_eps, 1);
    }
}
