// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Abs, Negative, Mask, MaskPosInf
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // BinaryMax
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"   // UnaryNe
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;       // input
    constexpr uint32_t cb_one = tt::CBIndex::c_1;     // one
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_2;  // mask_w
    CircularBuffer cb_one_obj(cb_one);
    CircularBuffer cb_mask_w_obj(cb_mask_w);

    constexpr uint32_t cb_y = tt::CBIndex::c_16;  // output

    constexpr uint32_t cb_val = tt::CBIndex::c_24;     // f(x)
    constexpr uint32_t cb_cal = tt::CBIndex::c_25;     // accumulator across cols
    constexpr uint32_t cb_reduce = tt::CBIndex::c_26;  // reduce output

    constexpr uint32_t onetile = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    cb_one_obj.wait_front(onetile);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;

    if (do_mask_w) {
        cb_mask_w_obj.wait_front(onetile);
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            // f(x) prologue — 2-branch dispatch on (do_mask_w && last-col).
            // Per-stage reconfig matches original *_with_dt calls.
            const bool mask_this = do_mask_w && (col_idx == Wt - 1);
            if (mask_this) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_x,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::CopyTile<
                        cb_mask_w,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::InputLifecycle::CallerManaged,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
#ifdef MINUS_INF
                    compute_kernel_lib::MaskPosInf<compute_kernel_lib::Dst::D0>{},
#else
                    compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
#endif
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
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            } else {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_x,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::InputLifecycle::Streaming,
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
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            }

            // Accumulator: col_idx==0 -> seed copy; else -> add (IS_ZERO) or max.
            if (col_idx == 0) {
                compute_kernel_lib::copy<
                    cb_val,
                    cb_cal,
                    compute_kernel_lib::CopyTileReconfig::Input,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>(onetile);
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
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::PackTile<
                        cb_cal,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#else
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_val,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::CopyTile<
                        cb_cal,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::BinaryMax<
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_cal,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#endif
            }
        }

        // Reduce f(x) across the row.
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
            cb_cal, cb_one, cb_reduce, compute_kernel_lib::ReduceInputBlockShape::single());

        // Final: copy reduce result -> [negate if MINUS_INF] -> cb_y.
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_reduce,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
#ifdef MINUS_INF
            compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
#endif
            compute_kernel_lib::PackTile<
                cb_y,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
    }

    cb_one_obj.pop_front(onetile);
    if (do_mask_w) {
        cb_mask_w_obj.pop_front(onetile);
    }
}
