// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused compute kernel: per-expert col-broadcast multiply, SFPU-accumulated.
//
// For each output tile we sum over reduction_dim_size experts:
//   dst_acc = Σ_e act_tile[e] * score_col[e]
//
// Each score tile has expert scores in column 0 (broadcast across columns).
// The reader pre-loads all reduction_dim_size score tiles before signalling
// compute_scores; they remain resident for the whole kernel.
//
// Accumulation note:
//   The LLK ELWMUL+COL MOP (tt_llk_*/llk_lib/llk_math_eltwise_binary.h)
//   hardcodes acc_to_dest=0, so mul_tiles_bcast_cols always overwrites its
//   destination dst slot — there is no hardware MAC path available here.
//   We therefore multiply each expert's contribution into a scratch dst slot
//   and accumulate into dst_acc via the SFPU add_binary_tile op.
//
// After tile_regs_acquire(), dst slots are hardware-zero-initialized, so
// dst_acc starts at 0 and the first add seeds the accumulator correctly.

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"

using namespace ckernel;

constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
constexpr uint32_t reduction_dim_size = get_compile_time_arg_val(1);
constexpr uint32_t input_granularity = get_compile_time_arg_val(2);
constexpr uint32_t compute_input_cb_id_0 = get_compile_time_arg_val(3);
constexpr uint32_t compute_input_cb_id_1 = get_compile_time_arg_val(4);
constexpr uint32_t compute_output_cb_id = get_compile_time_arg_val(5);

void kernel_main() {
    constexpr uint32_t dst_acc = 0;
    constexpr uint32_t dst_tmp = 1;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t num_input_tiles_iter = reduction_dim_size / input_granularity;

    // FPU MOP init for ELWMUL + COL-broadcast (mul_tiles_bcast_cols).
    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
        compute_input_cb_id_0, compute_input_cb_id_1, compute_output_cb_id);

    // SFPU init for dst-to-dst add (used to accumulate per-expert products).
    add_binary_tile_init();

    reconfig_data_format(compute_input_cb_id_0, compute_input_cb_id_1);

    // Wait for all score tiles — pre-loaded once by the reader prologue,
    // resident for the entire kernel invocation.
    cb_wait_front(compute_input_cb_id_1, reduction_dim_size);
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        tile_regs_acquire();

        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            cb_wait_front(compute_input_cb_id_0, input_granularity);

            for (uint32_t k = 0; k < input_granularity; ++k) {
                const uint32_t expert_tile = j * input_granularity + k;
                // dst_tmp = act_tile[k] * score_col[expert_tile]
                mul_tiles_bcast_cols(compute_input_cb_id_0, compute_input_cb_id_1, k, expert_tile, dst_tmp);
                // dst_acc += dst_tmp
                add_binary_tile(dst_acc, dst_tmp, dst_acc);
            }
            cb_pop_front(compute_input_cb_id_0, input_granularity);
        }

        tile_regs_commit();
        cb_reserve_back(compute_output_cb_id, one_tile);
        pack_reconfig_data_format(compute_output_cb_id);
        tile_regs_wait();
        pack_tile(dst_acc, compute_output_cb_id);
        tile_regs_release();
        cb_push_back(compute_output_cb_id, one_tile);
    }

    // Release all score tiles
    cb_pop_front(compute_input_cb_id_1, reduction_dim_size);
}
