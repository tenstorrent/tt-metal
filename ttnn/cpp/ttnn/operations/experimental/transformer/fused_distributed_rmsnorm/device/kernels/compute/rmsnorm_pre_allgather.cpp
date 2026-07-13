// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
 * For rmsnorm we compute E(x**2) and return it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/debug/dprint_pages.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_scalar_cb = get_compile_time_arg_val(1);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(4);
    constexpr uint32_t block_size = get_compile_time_arg_val(5);

    uint32_t num_tile_rows_to_process = get_arg_val<uint32_t>(0);
    constexpr uint32_t onetile = 1;

    binary_op_init_common(input_cb, input_cb, intermediate_cb);

    for (uint32_t tile_row_num = 0; tile_row_num < num_tile_rows_to_process; tile_row_num++) {
        /*
         * x**2
         */
        reconfig_data_format(input_cb, input_cb);
        pack_reconfig_data_format(intermediate_cb);

        cb_reserve_back(intermediate_cb, onetile);
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const uint32_t current_block_size =
                (col_tile + block_size <= num_tile_cols) ? block_size : (num_tile_cols - col_tile);
            cb_wait_front(input_cb, block_size);

            if (col_tile == 0) {
                // Seed the row accumulator by overwriting its single reserved tile.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::single(),
                    ckl::BinaryFpu<
                        input_cb,
                        input_cb,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::None>{},
                    ckl::PackTile<
                        intermediate_cb,
                        ckl::OutputLifecycle::CallerManaged,
                        ckl::PackTileReconfig::None,
                        ckl::Dst::D0,
                        ckl::TileOffset::Set>{0});

                if (current_block_size > 1) {
                    // Fold the rest of the first input block into that same output tile.
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(current_block_size - 1, current_block_size - 1),
                        ckl::BinaryFpu<
                            input_cb,
                            input_cb,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::None,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::None,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            ckl::OperandKind::Block,
                            ckl::TileOffset::Set,
                            ckl::TileOffset::Set>{1, 1},
                        ckl::PackTile<
                            intermediate_cb,
                            ckl::OutputLifecycle::L1AccumulationCallerManaged,
                            ckl::PackTileReconfig::None,
                            ckl::Dst::D0,
                            ckl::TileOffset::Unset,
                            ckl::PackTileL1Accumulation::Enabled>{});
                }
            } else {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(current_block_size, current_block_size),
                    ckl::BinaryFpu<
                        input_cb,
                        input_cb,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::None,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block>{},
                    ckl::PackTile<
                        intermediate_cb,
                        ckl::OutputLifecycle::L1AccumulationCallerManaged,
                        ckl::PackTileReconfig::None,
                        ckl::Dst::D0,
                        ckl::TileOffset::Unset,
                        ckl::PackTileL1Accumulation::Enabled>{});
            }

            cb_pop_front(input_cb, block_size);
        }
        cb_push_back(intermediate_cb, onetile);

        /*
         * sum(x**2)
         */
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, intermediate_cb, reduce_scalar_cb, output_cb>(
            compute_kernel_lib::ReduceInputBlockShape::single());
    }
    cb_pop_front(reduce_scalar_cb, onetile);
}
