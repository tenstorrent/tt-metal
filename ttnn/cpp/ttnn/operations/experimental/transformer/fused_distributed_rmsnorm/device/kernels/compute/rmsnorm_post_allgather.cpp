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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

namespace ckl = compute_kernel_lib;

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
        // ROPE tracking variables
        uint32_t rope_cos_tile_in_head = 0;
        uint32_t rope_sin_tile_in_head = 0;

        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x1**2), ...]
         * Uses auto-batched STREAMING mode - library handles CB lifecycle
         */
        ckl::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, stats_cb, reduce_scalar_cb, reduce_result_cb>(
            ckl::ReduceInputBlockShape::row(stats_tiles_cols));

        /*
         * 1/sqrt(mean_squared + eps)
         * PARTIAL migration: BinaryFpu(Add) + Rsqrt + PackTile, all on the same CB
         * (reduce_result_cb read for input, packed back as output).
         *
         * Reconfig audit: explicit reconfig_data_format + add_tiles_init -> Input.
         * Explicit pack_reconfig_data_format -> Output.
         * use_legacy_rsqrt forwarded via Legacy::On/Off.
         *
         * Lifecycle: reduce_result_cb InputLifecycle::Streaming (chain owns wait+pop); epsilon_cb
         * InputLifecycle::CallerManaged (waited once at MAIN entry); reduce_result_cb output
         * OutputLifecycle::Streaming (chain owns reserve+push). Same-CB in/out is fine — original
         * popped the input tile then re-reserved+packed; chain emits the same
         * sequence.
         */
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(reduce_result_cb),
                ckl::input(epsilon_cb, ckl::InputLifecycle::CallerManaged),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None>{},
            ckl::Rsqrt<ckl::Approx::Exact, use_legacy_rsqrt ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(reduce_result_cb)>{});

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */
        cb_wait_front(reduce_result_cb, 1);
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            ckl::mul<
                ckl::input(input_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(reduce_result_cb, ckl::InputLifecycle::CallerManaged),
                ckl::output(mul_rms_result_cb, ckl::OutputLifecycle::Bulk),
                ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(block_size, block_size));

            /**
             * Weight (gamma) fusion
             */
            if constexpr (has_weight) {
                cb_wait_front(weight_cb, col_tile + block_size);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size),
                    ckl::BinaryFpu<
                        ckl::input(mul_rms_result_cb, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
                        ckl::input(
                            weight_cb,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Set),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row>{0u, col_tile},
                    ckl::PackTile<ckl::output(mul_weight_result_cb, ckl::OutputLifecycle::Chunked)>{});
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
                matmul_init(intermediate_cb, transformation_mat_cb);
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

                ckl::add<
                    ckl::input(intermediate_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(rotated_input_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::output(output_cb, ckl::OutputLifecycle::Bulk),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(block_size, block_size));

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
