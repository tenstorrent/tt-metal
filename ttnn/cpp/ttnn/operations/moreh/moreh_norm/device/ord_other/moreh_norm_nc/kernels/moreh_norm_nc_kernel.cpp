// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Abs, Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // BinaryMax
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"   // UnaryNe
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto num_reduced_tiles_along_dim = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;    // input
    constexpr uint32_t cb_one = tt::CBIndex::c_1;  // one
    CircularBuffer cb_one_obj(cb_one);

    constexpr uint32_t cb_y = tt::CBIndex::c_16;  // output

    constexpr uint32_t cb_val = tt::CBIndex::c_24;
    constexpr uint32_t cb_cal = tt::CBIndex::c_25;

    constexpr uint32_t onetile = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    cb_one_obj.wait_front(onetile);

    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // f(x): no mask in nc path. IS_ZERO -> unary_ne; default -> abs.
            // MINUS_INF additionally negates.
            // Per-stage reconfig matches original *_with_dt calls.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::CopyTile<
                    cb_x,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>{},
#ifdef IS_ZERO
                compute_kernel_lib::UnaryNe<compute_kernel_lib::Dst::D0>{0u},
#else
                compute_kernel_lib::Abs<compute_kernel_lib::Dst::D0>{},
#endif
#ifdef MINUS_INF
                compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
#endif
                compute_kernel_lib::PackTile<
                    cb_val,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});

            // Accumulator over N/C dimension.
            if (inner_idx == 0) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_val,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::PackTile<
                        cb_cal,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            } else {
#ifdef IS_ZERO
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_val,
                        cb_cal,
                        compute_kernel_lib::BinaryFpuOp::Add,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::PackTile<
                        cb_cal,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#else
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_val,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::CopyTile<
                        cb_cal,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::BinaryMax<
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_cal,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutStreaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#endif
            }
        }

        // Final: copy cb_cal -> [negate if MINUS_INF] -> cb_y.
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_cal,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
#ifdef MINUS_INF
            compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
#endif
            compute_kernel_lib::PackTile<
                cb_y,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
    }
    cb_one_obj.pop_front(onetile);
}
