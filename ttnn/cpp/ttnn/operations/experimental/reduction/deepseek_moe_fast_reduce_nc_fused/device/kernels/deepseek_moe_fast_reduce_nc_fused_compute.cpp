// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused compute kernel: multiply-accumulate using hardware MAC.
//
// For each output tile we iterate over reduction_dim_size experts and sum:
//   dst0 += act_tile[e] * score_col[e]   (hardware MAC with col-broadcast)
//
// Each score tile has expert scores in column 0 (broadcast to all columns).
// The reader pre-loads all reduction_dim_size score tiles before signalling compute_scores.
// Score tiles are kept resident and accessed by index throughout the loop.
//
// Initialization:
//   init_bcast<ELWMUL, COL> handles PACK + UNPACK + hw_configure.
//   We then override the MATH init with acc_to_dest=1 so that every
//   mul_tiles_bcast_cols call accumulates into dst rather than replacing it.
//
// After tile_regs_acquire(), dst0 is zero-initialized by hardware, so the
// first MAC (expert 0) correctly seeds the accumulator.

#include "api/compute/bcast.h"

using namespace ckernel;

constexpr uint32_t num_output_tiles_to_process = get_compile_time_arg_val(0);
constexpr uint32_t reduction_dim_size = get_compile_time_arg_val(1);
constexpr uint32_t input_granularity = get_compile_time_arg_val(2);
constexpr uint32_t compute_input_cb_id_0 = get_compile_time_arg_val(3);
constexpr uint32_t compute_input_cb_id_1 = get_compile_time_arg_val(4);
constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(5);

void kernel_main() {
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t num_input_tiles_iter = reduction_dim_size / input_granularity;

    // Full PACK + UNPACK + hw_configure init for ELWMUL + COL-broadcast
    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
        compute_input_cb_id_0, compute_input_cb_id_1, compute_output_cb_id);

    // Override MATH init to enable acc_to_dest=1 (hardware accumulate mode)
    // This makes each mul_tiles_bcast_cols call do: dst0 += act * score  (MAC)
    MATH((llk_math_eltwise_binary_init_with_operands<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        compute_input_cb_id_0, compute_input_cb_id_1, 1 /*acc_to_dest*/)));

    // Wait for all score tiles — they are pre-loaded once by the reader prologue
    // and remain resident for the entire kernel invocation.
    cb_wait_front(compute_input_cb_id_1, reduction_dim_size);
    for (uint32_t i = 0; i < num_output_tiles_to_process; ++i) {
        tile_regs_acquire();

        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            cb_wait_front(compute_input_cb_id_0, input_granularity);
            reconfig_data_format(compute_input_cb_id_0, compute_input_cb_id_1);

            for (uint32_t k = 0; k < input_granularity; ++k) {
                // expert_tile = linear expert index for this tile
                const uint32_t expert_tile = j * input_granularity + k;
                // dst0 += act_tile[k] * score_col[expert_tile]  (single MAC)
                mul_tiles_bcast_cols(compute_input_cb_id_0, compute_input_cb_id_1, k, expert_tile, dst0);
            }
            cb_pop_front(compute_input_cb_id_0, input_granularity);
        }

        tile_regs_commit();
        cb_reserve_back(compute_output_cb_id, one_tile);
        pack_reconfig_data_format(compute_output_cb_id);
        tile_regs_wait();
        pack_tile(dst0, compute_output_cb_id);
        tile_regs_release();
        cb_push_back(compute_output_cb_id, one_tile);
    }

    // Release all score tiles
    cb_pop_front(compute_input_cb_id_1, reduction_dim_size);
}
