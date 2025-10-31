// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm, dependent on the RMSNORM define.
 * For rmsnorm we receive E(x**2) and compute the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) is contained in a one tile wide tensor containing E(x**2) in the left most column.
 */

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t stats_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight_cb = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_scalar_cb = get_compile_time_arg_val(3);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(4);
    constexpr uint32_t reduce_result_cb = get_compile_time_arg_val(5);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(6);
    constexpr uint32_t output_cb = get_compile_time_arg_val(7);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(8);
    constexpr uint32_t block_size = get_compile_time_arg_val(9);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(10);
    constexpr bool use_float32_reduction = get_compile_time_arg_val(11);
    constexpr bool use_legacy_rsqrt = get_compile_time_arg_val(12);
    constexpr uint32_t has_weight = get_compile_time_arg_val(13);

    const uint32_t num_tile_rows_to_process = get_arg_val<uint32_t>(0);
    binary_op_init_common(input_cb, input_cb, input_cb);

    cb_wait_front(reduce_scalar_cb, 1);  // comes from the reader
    cb_wait_front(epsilon_cb, 1);        // comes from the reader
    // if constexpr (has_weight) {
    //     cb_wait_front(weight_cb, num_tile_cols);
    // }

    /**
     * If there is a weight to apply, the result of x * RMS must be stored in an intermediate CB.
     * Otherwise, the result can be written directly to the output CB.
     */
    constexpr uint32_t mul_rms_result_cb = has_weight ? intermediate_cb : output_cb;

    for (uint32_t tile_row = 0; tile_row < num_tile_rows_to_process; tile_row++) {
        reconfig_data_format(stats_cb, reduce_scalar_cb);
        pack_reconfig_data_format(reduce_result_cb);

        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x0), sum(x1**2), sum(x1), ...]
         * RMSNorm packs mean(x**2) into cb_var. Layernorm just uses cb_stats_reduced.
         */
        reduce_init<REDUCE_OP, REDUCE_DIM, use_float32_reduction>(stats_cb, reduce_scalar_cb, reduce_result_cb);

        cb_wait_front(stats_cb, stats_tiles_cols);
        cb_reserve_back(reduce_result_cb, 1);

        tile_regs_acquire();
        // Reduce sum(x**2) first
        for (uint32_t i = 0; i < stats_tiles_cols; i++) {
            reduce_tile<REDUCE_OP, REDUCE_DIM, use_float32_reduction>(stats_cb, reduce_scalar_cb, i, 0, 0);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, reduce_result_cb);
        tile_regs_release();
        cb_push_back(reduce_result_cb, 1);
        cb_pop_front(stats_cb, stats_tiles_cols);

        reduce_uninit<false>();  // NOTE: cannot pass use_float32_reduction here or outputs are incorrect?

        /*
         * 1/sqrt(var + eps)
         */
        cb_wait_front(reduce_result_cb, 1);
        reconfig_data_format(reduce_result_cb, epsilon_cb);
        pack_reconfig_data_format(reduce_result_cb);

        add_tiles_init(reduce_result_cb, epsilon_cb);
        tile_regs_acquire();
        add_tiles(reduce_result_cb, epsilon_cb, 0, 0, 0);
        rsqrt_tile_init<use_legacy_rsqrt>();
        rsqrt_tile<use_legacy_rsqrt>(0);
        tile_regs_commit();
        cb_pop_front(reduce_result_cb, 1);
        cb_reserve_back(reduce_result_cb, 1);
        tile_regs_wait();
        pack_tile(0, reduce_result_cb);
        tile_regs_release();
        cb_push_back(reduce_result_cb, 1);

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */
        reconfig_data_format(input_cb, reduce_result_cb);
        pack_reconfig_data_format(mul_rms_result_cb);
        mul_bcast_cols_init_short(input_cb, reduce_result_cb);
        cb_wait_front(reduce_result_cb, 1);
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            cb_wait_front(input_cb, block_size);
            cb_reserve_back(mul_rms_result_cb, block_size);

            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                mul_tiles_bcast_cols(input_cb, reduce_result_cb, i, 0, i);
                pack_tile(i, mul_rms_result_cb);
            }
            tile_regs_commit();
            tile_regs_release();

            cb_push_back(mul_rms_result_cb, block_size);
            cb_pop_front(input_cb, block_size);

            if constexpr (has_weight) {
                // Reconfigure for mul_bcast_row
                reconfig_data_format(mul_rms_result_cb, weight_cb);
                pack_reconfig_data_format(output_cb);
                mul_bcast_rows_init_short(mul_rms_result_cb, weight_cb);
                // cumulative wait
                cb_wait_front(weight_cb, col_tile + block_size);
                cb_wait_front(mul_rms_result_cb, block_size);
                cb_reserve_back(output_cb, block_size);
                tile_regs_acquire();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    mul_tiles_bcast_rows(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
                    pack_tile(i, output_cb);
                }
                tile_regs_commit();
                tile_regs_release();
                cb_push_back(output_cb, block_size);
                cb_pop_front(mul_rms_result_cb, block_size);

                // Reconfigure for mul_bcast_col
                reconfig_data_format(input_cb, reduce_result_cb);
                pack_reconfig_data_format(mul_rms_result_cb);
                mul_bcast_cols_init_short(input_cb, reduce_result_cb);
            }
        }
        cb_pop_front(reduce_result_cb, 1);
    }
    cb_pop_front(epsilon_cb, 1);
    cb_pop_front(reduce_scalar_cb, 1);
    if constexpr (has_weight) {
        cb_pop_front(weight_cb, num_tile_cols);
    }
}
}  // namespace NAMESPACE
