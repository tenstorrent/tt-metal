// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Abs, Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/predicates.hpp"  // UnaryNe
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"     // Optional
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const auto num_output_tiles_per_core = get_arg(args::num_output_tiles_per_core);
    const auto num_reduced_tiles_along_dim = get_arg(args::num_reduced_tiles_along_dim);

    DataflowBuffer dfb_one_obj(dfb::one);

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::y);

    dfb_one_obj.wait_front(onetile);  // comes from the reader

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
    // Compute-private intermediates (dfb::val, dfb::cal): this kernel is their only toucher, so each is
    // self-looped on the host (bound PRODUCER and CONSUMER under one accessor name).
    for (uint32_t outer_idx = 0; outer_idx < num_output_tiles_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_reduced_tiles_along_dim; ++inner_idx) {
            // x != 0
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(onetile),
                ckl::CopyTile<ckl::input(dfb::x)>{},
                ckl::Optional<is_zero, ckl::UnaryNe<ckl::Dst::D0>>{0u},
                ckl::Optional<!is_zero, ckl::Abs<ckl::Dst::D0>>{},
                ckl::Optional<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(dfb::val)>{});

            // calculate f(x) over dimensions
            if (inner_idx == 0) {
                ckl::copy<ckl::input(dfb::val), ckl::output(dfb::cal)>(ckl::IterationShape::tiles(onetile));
            } else {
#ifdef IS_ZERO
                ckl::add<ckl::input(dfb::val), ckl::input(dfb::cal), ckl::output(dfb::cal)>(
                    ckl::IterationShape::tiles(onetile));
#else
                ckl::binary_sfpu<ckl::BinaryMax<>, ckl::input(dfb::val), ckl::input(dfb::cal), ckl::output(dfb::cal)>(
                    ckl::IterationShape::tiles(onetile));
#endif
            }
        }

        // Compute dfb::y
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::CopyTile<ckl::input(dfb::cal)>{},
            ckl::Optional<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(dfb::y)>{});
    }
    dfb_one_obj.pop_front(onetile);
}
