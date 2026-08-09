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
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t input_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t stats_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t weight_dfb_id = get_compile_time_arg_val(2);
    constexpr uint32_t reduce_scalar_dfb_id = get_compile_time_arg_val(3);
    constexpr uint32_t epsilon_dfb_id = get_compile_time_arg_val(4);
    constexpr uint32_t reduce_result_dfb_id = get_compile_time_arg_val(5);
    constexpr uint32_t intermediate_dfb_id = get_compile_time_arg_val(6);
    constexpr uint32_t output_dfb_id = get_compile_time_arg_val(7);
    constexpr uint32_t transformation_mat_dfb_id = get_compile_time_arg_val(8);
    constexpr uint32_t rope_cos_dfb_id = get_compile_time_arg_val(9);
    constexpr uint32_t rope_sin_dfb_id = get_compile_time_arg_val(10);
    constexpr uint32_t rotated_input_dfb_id = get_compile_time_arg_val(11);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(12);
    constexpr uint32_t block_size = get_compile_time_arg_val(13);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(14);
    constexpr bool use_legacy_rsqrt = get_compile_time_arg_val(15);
    constexpr uint32_t has_weight = get_compile_time_arg_val(16);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(17);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(18);

    const uint32_t num_tile_rows_to_process = get_arg_val<uint32_t>(0);

    DataflowBuffer dfb_reduce_scalar(reduce_scalar_dfb_id);
    DataflowBuffer dfb_epsilon(epsilon_dfb_id);
    DataflowBuffer dfb_transformation_mat(transformation_mat_dfb_id);
    DataflowBuffer dfb_reduce_result(reduce_result_dfb_id);
    DataflowBuffer dfb_input(input_dfb_id);
    DataflowBuffer dfb_weight(weight_dfb_id);
    DataflowBuffer dfb_intermediate(intermediate_dfb_id);
    DataflowBuffer dfb_rotated_input(rotated_input_dfb_id);
    DataflowBuffer dfb_rope_cos(rope_cos_dfb_id);
    DataflowBuffer dfb_rope_sin(rope_sin_dfb_id);
    DataflowBuffer dfb_output(output_dfb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(intermediate_dfb_id, transformation_mat_dfb_id, rotated_input_dfb_id);
    matmul_init(intermediate_dfb_id, transformation_mat_dfb_id);

    compute_kernel_hw_startup(input_dfb_id, input_dfb_id, input_dfb_id);

    dfb_reduce_scalar.wait_front(1);  // comes from the reader
    dfb_epsilon.wait_front(1);        // comes from the reader
    if constexpr (fuse_rope) {
        dfb_transformation_mat.wait_front(1);
    }

    /**
     * If there is a weight to apply (or if ROPE is fused), the result of x * RMS must be stored in an intermediate DFB.
     * Otherwise, the result can be written directly to the output DFB.
     * When applying the weight, the result of x * weight must be stored in an intermediate DFB if ROPE is fused,
     * otherwise it can be written directly to the output DFB.
     */
    constexpr uint32_t mul_rms_result_dfb_id = (fuse_rope || has_weight) ? intermediate_dfb_id : output_dfb_id;
    constexpr uint32_t mul_weight_result_dfb_id = fuse_rope ? intermediate_dfb_id : output_dfb_id;
    DataflowBuffer dfb_mul_rms_result(mul_rms_result_dfb_id);
    DataflowBuffer dfb_mul_weight_result(mul_weight_result_dfb_id);

    for (uint32_t tile_row = 0; tile_row < num_tile_rows_to_process; tile_row++) {
        // ROPE tracking variables
        uint32_t rope_cos_tile_in_head = 0;
        uint32_t rope_sin_tile_in_head = 0;

        /*
         * Reduce stats input.
         * dfb_stats_id = [sum(x0**2), sum(x1**2), ...]
         * Uses auto-batched STREAMING mode - library handles DFB lifecycle
         */
        ckl::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, stats_dfb_id, reduce_scalar_dfb_id, reduce_result_dfb_id>(
            ckl::ReduceInputBlockShape::row(stats_tiles_cols));

        // 1/sqrt(mean_squared + eps)
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(reduce_result_dfb_id),
                ckl::input(epsilon_dfb_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None>{},
            ckl::Rsqrt<ckl::Approx::Exact, use_legacy_rsqrt ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(reduce_result_dfb_id)>{});

        /*
         * norm x
         * RMSNorm: X * 1/sqrt(E[X**2] + eps)
         */
        dfb_reduce_result.wait_front(1);
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            ckl::mul<
                ckl::input(input_dfb_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(reduce_result_dfb_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                ckl::output(mul_rms_result_dfb_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd),
                ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(block_size, block_size));

            /**
             * Weight (gamma) fusion
             */
            if constexpr (has_weight) {
                // cumulative wait
                dfb_weight.wait_front(col_tile + block_size);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(block_size, /*block_size=*/block_size),
                    ckl::BinaryFpu<
                        ckl::input(
                            mul_rms_result_dfb_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::OperandKind::Block),
                        ckl::input(
                            weight_dfb_id,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Set),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row>{0u, col_tile},
                    ckl::PackTile<ckl::output(
                        mul_weight_result_dfb_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
            }

            /**
             * ROPE fusion
             */
            if constexpr (fuse_rope) {
                /**
                 * Rotate the input, write to rotated_input_dfb_id
                 */
                reconfig_data_format(transformation_mat_dfb_id, intermediate_dfb_id);
                pack_reconfig_data_format(rotated_input_dfb_id);
                matmul_init(intermediate_dfb_id, transformation_mat_dfb_id);
                dfb_intermediate.wait_front(block_size);
                dfb_rotated_input.reserve_back(block_size);
                tile_regs_acquire();
                tile_regs_wait();

                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    matmul_tiles(intermediate_dfb_id, transformation_mat_dfb_id, i, 0, i);
                    pack_tile(i, rotated_input_dfb_id);
                }

                tile_regs_commit();
                tile_regs_release();
                dfb_rotated_input.push_back(block_size);

                /**
                 * Write x * cos in-place to mul_rms_result_dfb_id (intermediate_dfb_id)
                 */
                reconfig_data_format(intermediate_dfb_id, rope_cos_dfb_id);
                pack_reconfig_data_format(intermediate_dfb_id);
                mul_init(intermediate_dfb_id, rope_cos_dfb_id);
                dfb_rope_cos.wait_front(head_dim_tiles);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    mul_tiles(intermediate_dfb_id, rope_cos_dfb_id, i, rope_cos_tile_in_head, i);
                    rope_cos_tile_in_head++;
                    if (rope_cos_tile_in_head == head_dim_tiles) {
                        // Stride heads, reset the index
                        rope_cos_tile_in_head = 0;
                    }
                }
                tile_regs_commit();
                // Write in-place to intermediate_dfb_id
                dfb_intermediate.pop_front(block_size);
                dfb_intermediate.reserve_back(block_size);
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    pack_tile(i, intermediate_dfb_id);
                }
                tile_regs_release();
                dfb_intermediate.push_back(block_size);

                /**
                 * Write x_rotated * sin in-place to rotated_input_dfb_id
                 */
                reconfig_data_format(rotated_input_dfb_id, rope_sin_dfb_id);
                pack_reconfig_data_format(rotated_input_dfb_id);
                mul_init(rotated_input_dfb_id, rope_sin_dfb_id);
                dfb_rope_sin.wait_front(head_dim_tiles);
                dfb_rotated_input.wait_front(block_size);

                tile_regs_acquire();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    mul_tiles(rotated_input_dfb_id, rope_sin_dfb_id, i, rope_sin_tile_in_head, i);
                    rope_sin_tile_in_head++;
                    if (rope_sin_tile_in_head == head_dim_tiles) {
                        // Stride heads, reset the index
                        rope_sin_tile_in_head = 0;
                    }
                }
                tile_regs_commit();
                // Write in-place to rotated_input_dfb_id
                dfb_rotated_input.pop_front(block_size);
                dfb_rotated_input.reserve_back(block_size);
                tile_regs_wait();
                for (uint32_t i = 0; i < block_size && col_tile + i < num_tile_cols; i++) {
                    pack_tile(i, rotated_input_dfb_id);
                }
                tile_regs_release();
                dfb_rotated_input.push_back(block_size);

                ckl::add<
                    ckl::input(
                        intermediate_dfb_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                    ckl::input(
                        rotated_input_dfb_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                    ckl::output(output_dfb_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(block_size, block_size));

                // Reconfigure for mul_bcast_col
                reconfig_data_format(input_dfb_id, reduce_result_dfb_id);
                pack_reconfig_data_format(mul_rms_result_dfb_id);
                mul_bcast_cols_init(input_dfb_id, reduce_result_dfb_id);
            }
        }
        dfb_reduce_result.pop_front(1);

        if constexpr (fuse_rope) {
            // We have processed an entire row, so free up the rope cos/sin DFBs
            dfb_rope_cos.pop_front(head_dim_tiles);
            dfb_rope_sin.pop_front(head_dim_tiles);
        }
    }
    dfb_epsilon.pop_front(1);
    dfb_reduce_scalar.pop_front(1);
    if constexpr (has_weight) {
        dfb_weight.pop_front(num_tile_cols);
    }
    if constexpr (fuse_rope) {
        dfb_transformation_mat.pop_front(1);
    }
}
