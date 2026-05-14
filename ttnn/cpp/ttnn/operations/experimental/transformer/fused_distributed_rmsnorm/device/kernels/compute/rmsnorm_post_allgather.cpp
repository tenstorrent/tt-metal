// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm, dependent on the RMSNORM define.
 * For rmsnorm we receive E(x**2) and compute the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) is contained in a one tile wide tensor containing E(x**2) in the left most column.
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
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
    constexpr bool use_legacy_rsqrt = get_compile_time_arg_val(15);
    constexpr uint32_t has_weight = get_compile_time_arg_val(16);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(17);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(18);
    // Per-head mode: stats has `num_heads` tile cols per row (one per head, no AG); reduce_factor
    // is head_dim; we loop over heads and normalize each head's head_dim_tiles slice with its own
    // per-head RMS. When false (default = legacy/WAN behavior), stats has stats_tiles_cols tile
    // cols (post-AG global) and we reduce them all into one RMS applied across the whole row.
    constexpr bool per_head_norm = get_compile_time_arg_val(19);
    constexpr uint32_t num_heads = get_compile_time_arg_val(20);

    // Branchless legacy/per-head loop bounds. Legacy: 1 head iter covering num_tile_cols.
    constexpr uint32_t heads_per_row = per_head_norm ? num_heads : 1;
    constexpr uint32_t cols_per_head = per_head_norm ? head_dim_tiles : num_tile_cols;

    const uint32_t num_tile_rows_to_process = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(intermediate_cb, transformation_mat_cb, rotated_input_cb);
    matmul_init(intermediate_cb, transformation_mat_cb);

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
        // ROPE tracking variables — reset per row; per_head_norm mode also resets per-head
        // inside the loop below since each head's rope window starts at 0.
        uint32_t rope_cos_tile_in_head = 0;
        uint32_t rope_sin_tile_in_head = 0;

        for (uint32_t head_idx = 0; head_idx < heads_per_row; head_idx++) {
            // Per-head mode: each head starts a fresh rope window. (For legacy mode this is
            // a no-op since heads_per_row == 1 and the variable wraps naturally inside the
            // col loop via the original `if (rope_*_tile_in_head == head_dim_tiles) reset`.)
            if constexpr (per_head_norm) {
                rope_cos_tile_in_head = 0;
                rope_sin_tile_in_head = 0;
            }

            /*
             * Reduce stats input.
             *   Legacy: cb_stats has stats_tiles_cols tiles per row (gathered across devices),
             *           collapse them into one global sum-of-squares mean.
             *   Per-head: cb_stats has num_heads tiles per row, consume ONE per head iteration,
             *             yielding a per-head mean of squares.
             * The reduce_scalar baked into reduce_scalar_cb is `1/reduce_factor` set by the
             * program factory (= 1/(W*num_devices) for legacy, 1/head_dim for per-head).
             */
            if constexpr (per_head_norm) {
                compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, stats_cb, reduce_scalar_cb, reduce_result_cb>(compute_kernel_lib::ReduceInputBlockShape::single());
            } else {
                /*
                 * Reduce stats input.
                 * cb_stats = [sum(x0**2), sum(x1**2), ...]
                 * Uses auto-batched STREAMING mode - library handles CB lifecycle
                 */
                compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, stats_cb, reduce_scalar_cb, reduce_result_cb>(
                    compute_kernel_lib::ReduceInputBlockShape::row(stats_tiles_cols));
            }

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
             *
             * Per-head mode iterates `cols_per_head = head_dim_tiles` per head; weight_cb expects
             * a CUMULATIVE wait offset across heads (= head_idx * head_dim_tiles + col_tile).
             */
            reconfig_data_format(input_cb, reduce_result_cb);
            pack_reconfig_data_format(mul_rms_result_cb);
            mul_bcast_cols_init_short(input_cb, reduce_result_cb);
            cb_wait_front(reduce_result_cb, 1);
            const uint32_t head_col_base = head_idx * cols_per_head;
            for (uint32_t col_tile = 0; col_tile < cols_per_head; col_tile += block_size) {
                cb_wait_front(input_cb, block_size);
                cb_reserve_back(mul_rms_result_cb, block_size);

                tile_regs_acquire();
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
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
                    // cumulative wait — counts ALL weight tiles consumed so far in this row
                    // (across previous heads + this head's progress so far).
                    cb_wait_front(weight_cb, head_col_base + col_tile + block_size);
                    cb_wait_front(mul_rms_result_cb, block_size);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
                        mul_tiles_bcast_rows(mul_rms_result_cb, weight_cb, i, head_col_base + col_tile + i, i);
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
                    for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
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

                    for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
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
                    for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
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
                    for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
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
                    for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
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
                    for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
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
                    for (uint32_t i = 0; i < block_size && col_tile + i < cols_per_head; i++) {
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
        }  // end of head loop

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
