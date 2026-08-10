// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Abs, Negative, Mask, MaskPosInf
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/predicates.hpp"  // UnaryNe
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"     // Optional
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto num_cols_per_core = get_arg(args::num_cols_per_core);
    const auto Ht = get_arg(args::Ht);
    const auto origin_h = get_arg(args::origin_h);

    constexpr uint32_t dfb_x_id = dfb::x;
    constexpr uint32_t dfb_one_id = dfb::one;
    constexpr uint32_t dfb_mask_h_id = dfb::mask_h;
    DataflowBuffer dfb_one_obj(dfb_one_id);
    DataflowBuffer dfb_mask_h_obj(dfb_mask_h_id);

    constexpr uint32_t dfb_y_id = dfb::y;

    constexpr uint32_t dfb_val_id = dfb::val;
    constexpr uint32_t dfb_cal_id = dfb::cal;
    constexpr uint32_t dfb_reduce_id = dfb::reduce;

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::y);

    dfb_one_obj.wait_front(onetile);

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;

    if (do_mask_h) {
        dfb_mask_h_obj.wait_front(onetile);
    }

#ifdef MINUS_INF
    constexpr bool minus_inf = true;
#else
    constexpr bool minus_inf = false;
#endif
#ifdef IS_ZERO
    constexpr bool is_zero = true;
#else
    constexpr bool is_zero = false;
#endif
    using MaskOp =
        std::conditional_t<minus_inf, ckl::MaskPosInf<ckl::Dst::D0>, ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>>;
    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            const bool mask_this = do_mask_h && (row_idx == Ht - 1);
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(onetile),
                ckl::CopyTile<ckl::input(dfb_x_id)>{},
                ckl::runtime_if(
                    mask_this,
                    ckl::CopyTile<
                        ckl::input(dfb_mask_h_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                        ckl::Dst::D1>{},
                    MaskOp{}),
                ckl::Optional<is_zero, ckl::UnaryNe<ckl::Dst::D0>>{0u},
                ckl::Optional<!is_zero, ckl::Abs<ckl::Dst::D0>>{},
                ckl::Optional<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(dfb_val_id)>{});

            if (row_idx == 0) {
                ckl::copy<ckl::input(dfb_val_id), ckl::output(dfb_cal_id)>(ckl::IterationShape::tiles(onetile));
            } else {
#ifdef IS_ZERO
                ckl::add<ckl::input(dfb_val_id), ckl::input(dfb_cal_id), ckl::output(dfb_cal_id)>(
                    ckl::IterationShape::tiles(onetile));
#else
                ckl::binary_sfpu<
                    ckl::BinaryMax<>,
                    ckl::input(dfb_val_id),
                    ckl::input(dfb_cal_id),
                    ckl::output(dfb_cal_id)>(ckl::IterationShape::tiles(onetile));
#endif
            }
        }

        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_cal_id, dfb_one_id, dfb_reduce_id>(ckl::ReduceInputBlockShape::single());

        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::CopyTile<ckl::input(dfb_reduce_id)>{},
            ckl::Optional<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(dfb_y_id)>{});
    }

    dfb_one_obj.pop_front(onetile);
    if (do_mask_h) {
        dfb_mask_h_obj.pop_front(onetile);
    }
}
