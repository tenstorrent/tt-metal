// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Abs, Negative, Mask, MaskPosInf
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"  // UnaryNe
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"    // OptionalChainElement
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto Wt = get_arg(args::Wt);
    const auto origin_w = get_arg(args::origin_w);

    constexpr uint32_t cb_x = dfb::x;
    constexpr uint32_t cb_one = dfb::one;
    constexpr uint32_t cb_mask_w = dfb::mask_w;
    DataflowBuffer dfb_one_obj(cb_one);
    DataflowBuffer dfb_mask_w_obj(cb_mask_w);

    constexpr uint32_t cb_y = dfb::y;

    constexpr uint32_t cb_val = dfb::val;
    constexpr uint32_t cb_cal = dfb::cal;
    constexpr uint32_t cb_reduce = dfb::reduce;

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::y);

    dfb_one_obj.wait_front(onetile);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;

    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
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
    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const bool mask_this = do_mask_w && (col_idx == Wt - 1);
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_x)>{},
                ckl::runtime_if(
                    mask_this,
                    ckl::CopyTile<ckl::input(cb_mask_w, ckl::WaitPolicy::None, ckl::PopPolicy::None), ckl::Dst::D1>{},
                    MaskOp{}),
                ckl::OptionalChainElement<is_zero, ckl::UnaryNe<ckl::Dst::D0>>{0u},
                ckl::OptionalChainElement<!is_zero, ckl::Abs<ckl::Dst::D0>>{},
                ckl::OptionalChainElement<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(cb_val)>{});

            if (col_idx == 0) {
                ckl::copy<ckl::input(cb_val), ckl::output(cb_cal)>(ckl::EltwiseShape::tiles(onetile));
            } else {
#ifdef IS_ZERO
                ckl::add<ckl::input(cb_val), ckl::input(cb_cal), ckl::output(cb_cal)>(
                    ckl::EltwiseShape::tiles(onetile));
#else
                ckl::binary_sfpu<ckl::BinaryMax<>, ckl::input(cb_val), ckl::input(cb_cal), ckl::output(cb_cal)>(
                    ckl::EltwiseShape::tiles(onetile));
#endif
            }
        }

        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_cal, cb_one, cb_reduce>(ckl::ReduceInputBlockShape::single());

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<ckl::input(cb_reduce)>{},
            ckl::OptionalChainElement<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(cb_y)>{});
    }

    dfb_one_obj.pop_front(onetile);
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
