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

namespace ckl = compute_kernel_lib;

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
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<cb_x>{},
                    ckl::CopyTile<cb_mask_w, ckl::Dst::D1, ckl::InputLifecycle::CallerManaged>{},
#ifdef MINUS_INF
                    ckl::MaskPosInf<ckl::Dst::D0>{},
#else
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
#endif
#ifdef IS_ZERO
                    ckl::UnaryNe<ckl::Dst::D0>{0u},
#else
                    ckl::Abs<ckl::Dst::D0>{},
#endif
#ifdef MINUS_INF
                    ckl::Negative<ckl::Dst::D0>{},
#endif
                    ckl::PackTile<cb_val>{});
            } else {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<cb_x>{},
#ifdef IS_ZERO
                    ckl::UnaryNe<ckl::Dst::D0>{0u},
#else
                    ckl::Abs<ckl::Dst::D0>{},
#endif
#ifdef MINUS_INF
                    ckl::Negative<ckl::Dst::D0>{},
#endif
                    ckl::PackTile<cb_val>{});
            }

            // Accumulator: col_idx==0 -> seed copy; else -> add (IS_ZERO) or max.
            if (col_idx == 0) {
                ckl::copy<cb_val, cb_cal>(ckl::EltwiseShape::tiles(onetile));
            } else {
#ifdef IS_ZERO
                ckl::add<cb_val, cb_cal, cb_cal>(ckl::EltwiseShape::tiles(onetile));
#else
                ckl::binary_sfpu<ckl::BinaryMax<>, cb_val, cb_cal, cb_cal>(ckl::EltwiseShape::tiles(onetile));
#endif
            }
        }

        // Reduce f(x) across the row.
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_cal, cb_one, cb_reduce>(ckl::ReduceInputBlockShape::single());

        // Final: copy reduce result -> [negate if MINUS_INF] -> cb_y.
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<cb_reduce>{},
#ifdef MINUS_INF
            ckl::Negative<ckl::Dst::D0>{},
#endif
            ckl::PackTile<cb_y>{});
    }

    cb_one_obj.pop_front(onetile);
    if (do_mask_w) {
        cb_mask_w_obj.pop_front(onetile);
    }
}
