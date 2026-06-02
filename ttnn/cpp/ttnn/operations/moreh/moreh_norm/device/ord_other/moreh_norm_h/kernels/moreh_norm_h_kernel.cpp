// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Abs, Negative, Mask, MaskPosInf
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // BinaryMax
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"   // UnaryNe
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;       // input
    constexpr uint32_t cb_one = tt::CBIndex::c_1;     // one
    constexpr uint32_t cb_mask_h = tt::CBIndex::c_2;  // mask_h
    CircularBuffer cb_one_obj(cb_one);
    CircularBuffer cb_mask_h_obj(cb_mask_h);

    constexpr uint32_t cb_y = tt::CBIndex::c_16;  // output

    constexpr uint32_t cb_val = tt::CBIndex::c_24;     // f(x)
    constexpr uint32_t cb_cal = tt::CBIndex::c_25;     // accumulator (sum or max across rows)
    constexpr uint32_t cb_reduce = tt::CBIndex::c_26;  // reduce output

    constexpr uint32_t onetile = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    cb_one_obj.wait_front(onetile);

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;

    if (do_mask_h) {
        cb_mask_h_obj.wait_front(onetile);
    }

    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            // f(x) prologue. 2-branch dispatch on (do_mask_h && last-row).
            // Per-stage reconfig matches original copy_tile_init_with_dt / pack_tile_with_dt.
            // cb_x InputLifecycle::Streaming; cb_mask_h InputLifecycle::CallerManaged + Scalar (held outside loop).
            const bool mask_this = do_mask_h && (row_idx == Ht - 1);
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
                        cb_mask_h,
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
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            }

            // Accumulator: row_idx==0 -> seed copy; else -> reduce-across-rows op.
            //   IS_ZERO  -> add_tiles (sum of != zero count)
            //   default  -> binary_max (running max via two-DEST SFPU)
            if (row_idx == 0) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::CopyTile<
                        cb_val,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::CopyTileReconfig::Input>{},
                    compute_kernel_lib::PackTile<
                        cb_cal,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            } else {
#ifdef IS_ZERO
                // cb_cal = cb_val + cb_cal (in-place accumulator).
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
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#else
                // cb_cal = max(cb_val, cb_cal) via two-DEST SFPU.
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
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
#endif
            }
        }

        // Reduce f(x) accumulator across the column. Uses reduce helper.
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
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
    }

    cb_one_obj.pop_front(onetile);
    if (do_mask_h) {
        cb_mask_h_obj.pop_front(onetile);
    }
}
