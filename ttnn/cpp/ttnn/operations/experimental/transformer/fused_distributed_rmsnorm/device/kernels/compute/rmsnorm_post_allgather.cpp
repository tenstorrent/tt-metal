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
#include "compute_kernel_api/matmul.h"

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
    constexpr uint32_t transformation_mat_cb = get_compile_time_arg_val(8);
    constexpr uint32_t rope_cos_cb = get_compile_time_arg_val(9);
    constexpr uint32_t rope_sin_cb = get_compile_time_arg_val(10);
    constexpr uint32_t rotated_input_cb = get_compile_time_arg_val(11);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(12);
    constexpr uint32_t block_size = get_compile_time_arg_val(13);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(14);
    constexpr bool use_float32_reduction = get_compile_time_arg_val(15);
    constexpr bool use_legacy_rsqrt = get_compile_time_arg_val(16);
    constexpr uint32_t has_weight = get_compile_time_arg_val(17);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(18);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(19);

    const uint32_t num_tile_rows_to_process = get_arg_val<uint32_t>(0);
    mm_init(intermediate_cb, transformation_mat_cb, rotated_input_cb);

    binary_op_init_common(input_cb, input_cb, input_cb);

    cb_wait_front(reduce_scalar_cb, 1);  // comes from the reader
    cb_wait_front(epsilon_cb, 1);        // comes from the reader
    if constexpr (fuse_rope) {
        cb_wait_front(transformation_mat_cb, 1);
    }

    /**
     * If there is a weight to apply (or if ROPE is fused), the result of x * RMS must be stored in an intermediate CB.
     * Otherwise, the result can be written directly to the output CB.
     * When applying the weight, the result of x * weight must be stored in an intermediate CB if ROPE is fused,
     * otherwise it can be written directly to the output CB.
     */
    constexpr uint32_t mul_rms_result_cb = (fuse_rope || has_weight) ? intermediate_cb : output_cb;
    constexpr uint32_t mul_weight_result_cb = fuse_rope ? intermediate_cb : output_cb;

    for (uint32_t tile_row = 0; tile_row < num_tile_rows_to_process; tile_row++) {
        // ROPE tracking variables
        uint32_t rope_cos_tile_in_head = 0;
        uint32_t rope_sin_tile_in_head = 0;

        reconfig_data_format(stats_cb, reduce_scalar_cb);
        pack_reconfig_data_format(reduce_result_cb);

        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x1**2), ...]
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
         * 1/sqrt(mean_squared + eps)
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

            /**
             * Weight (gamma) fusion
             */
            if constexpr (has_weight) {
                // Reconfigure for mul_bcast_row
                reconfig_data_format(mul_rms_result_cb, weight_cb);
                pack_reconfig_data_format(mul_weight_result_cb);
                mul_bcast_rows_init_short(mul_rms_result_cb, weight_cb);
                // cumulative wait
                cb_wait_front(weight_cb, col_tile + block_size);
                cb_wait_front(mul_rms_result_cb, block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    mul_tiles_bcast_rows(mul_rms_result_cb, weight_cb, i, col_tile + i, i);
                }
                tile_regs_commit();

                /**
                 * The compute loop must be written like this because if rope is fused,
                 * mul_weight_result_cb == mul_rms_result_cb
                 * and so this is an in-place operation.
                 * If rope is not fused, mul_weight_result_cb == output_cb
                 */
                cb_pop_front(mul_rms_result_cb, block_size);
                cb_reserve_back(mul_weight_result_cb, block_size);

                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    pack_tile(i, mul_weight_result_cb);
                }
                tile_regs_release();
                cb_push_back(mul_weight_result_cb, block_size);

                // Reconfigure for mul_bcast_col
                reconfig_data_format(input_cb, reduce_result_cb);
                pack_reconfig_data_format(mul_rms_result_cb);
                mul_bcast_cols_init_short(input_cb, reduce_result_cb);
            }

            /**
             * ROPE fusion
             */
            if constexpr (fuse_rope) {
                /**
                 * Rotate the input, write to rotated_input_cb
                 */
                reconfig_data_format(transformation_mat_cb, intermediate_cb);
                pack_reconfig_data_format(rotated_input_cb);
                mm_init_short(intermediate_cb, transformation_mat_cb);
                cb_wait_front(intermediate_cb, block_size);
                cb_reserve_back(rotated_input_cb, block_size);
                tile_regs_acquire();
                tile_regs_wait();

                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    matmul_tiles(intermediate_cb, transformation_mat_cb, i, 0, i);
                    pack_tile(i, rotated_input_cb);
                }

                tile_regs_commit();
                tile_regs_release();
                cb_push_back(rotated_input_cb, block_size);

                /**
                 * Write x * cos in-place to mul_rms_result_cb (intermediate_cb)
                 */
                reconfig_data_format(intermediate_cb, rope_cos_cb);
                pack_reconfig_data_format(intermediate_cb);
                mul_tiles_init(intermediate_cb, rope_cos_cb);
                cb_wait_front(rope_cos_cb, head_dim_tiles);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    mul_tiles(intermediate_cb, rope_cos_cb, i, rope_cos_tile_in_head, i);
                    rope_cos_tile_in_head++;
                    if (rope_cos_tile_in_head == head_dim_tiles) {
                        // Stride heads, reset the index
                        rope_cos_tile_in_head = 0;
                    }
                }
                tile_regs_commit();
                // Write in-place to intermediate_cb
                cb_pop_front(intermediate_cb, block_size);
                cb_reserve_back(intermediate_cb, block_size);
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    pack_tile(i, intermediate_cb);
                }
                tile_regs_release();
                cb_push_back(intermediate_cb, block_size);

                /**
                 * Write x_rotated * sin in-place to rotated_input_cb
                 */
                reconfig_data_format(rotated_input_cb, rope_sin_cb);
                pack_reconfig_data_format(rotated_input_cb);
                mul_tiles_init(rotated_input_cb, rope_sin_cb);
                cb_wait_front(rope_sin_cb, head_dim_tiles);
                cb_wait_front(rotated_input_cb, block_size);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    mul_tiles(rotated_input_cb, rope_sin_cb, i, rope_sin_tile_in_head, i);
                    rope_sin_tile_in_head++;
                    if (rope_sin_tile_in_head == head_dim_tiles) {
                        // Stride heads, reset the index
                        rope_sin_tile_in_head = 0;
                    }
                }
                tile_regs_commit();
                // Write in-place to rotated_input_cb
                cb_pop_front(rotated_input_cb, block_size);
                cb_reserve_back(rotated_input_cb, block_size);
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    pack_tile(i, rotated_input_cb);
                }
                tile_regs_release();
                cb_push_back(rotated_input_cb, block_size);

                /**
                 * Write cos_interim + sin_interim to output_cb
                 */
                reconfig_data_format(intermediate_cb, rotated_input_cb);
                pack_reconfig_data_format(output_cb);
                add_tiles_init(intermediate_cb, rotated_input_cb);
                cb_wait_front(intermediate_cb, block_size);
                cb_wait_front(rotated_input_cb, block_size);
                cb_reserve_back(output_cb, block_size);

                tile_regs_acquire();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    add_tiles(intermediate_cb, rotated_input_cb, i, i, i);
                    pack_tile(i, output_cb);
                }
                tile_regs_commit();
                tile_regs_release();
                cb_push_back(output_cb, block_size);

                cb_pop_front(intermediate_cb, block_size);
                cb_pop_front(rotated_input_cb, block_size);

                // Reconfigure for mul_bcast_col
                reconfig_data_format(input_cb, reduce_result_cb);
                pack_reconfig_data_format(mul_rms_result_cb);
                mul_bcast_cols_init_short(input_cb, reduce_result_cb);
            }
        }
        cb_pop_front(reduce_result_cb, 1);

        if constexpr (fuse_rope) {
            // We have processed an entire row, so free up the rope cos/sin CBs
            cb_pop_front(rope_cos_cb, head_dim_tiles);
            cb_pop_front(rope_sin_cb, head_dim_tiles);
        }
    }
    cb_pop_front(epsilon_cb, 1);
    cb_pop_front(reduce_scalar_cb, 1);
    if constexpr (has_weight) {
        cb_pop_front(weight_cb, num_tile_cols);
    }
    if constexpr (fuse_rope) {
        cb_pop_front(transformation_mat_cb, 1);
    }
}
}  // namespace NAMESPACE
