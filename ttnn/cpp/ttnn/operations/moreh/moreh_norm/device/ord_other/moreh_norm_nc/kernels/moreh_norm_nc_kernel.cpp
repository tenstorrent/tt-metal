// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Abs, Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_predicates.hpp"  // UnaryNe
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"    // OptionalChainElement
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto num_output_tiles_per_core = get_arg(args::num_output_tiles_per_core);
    const auto num_reduced_tiles_along_dim = get_arg(args::num_reduced_tiles_along_dim);

    constexpr uint32_t cb_x = dfb::x;
    constexpr uint32_t cb_one = dfb::one;
    DataflowBuffer dfb_one_obj(cb_one);

    constexpr uint32_t cb_y = dfb::y;

    constexpr uint32_t cb_val = dfb::val;
    constexpr uint32_t cb_cal = dfb::cal;

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::y);

    dfb_one_obj.wait_front(onetile);

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
    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<ckl::input(cb_x)>{},
                ckl::OptionalChainElement<is_zero, ckl::UnaryNe<ckl::Dst::D0>>{0u},
                ckl::OptionalChainElement<!is_zero, ckl::Abs<ckl::Dst::D0>>{},
                ckl::OptionalChainElement<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(cb_val)>{});

            if (inner_idx == 0) {
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

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<ckl::input(cb_cal)>{},
            ckl::OptionalChainElement<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(cb_y)>{});
    }
    dfb_one_obj.pop_front(onetile);
}
