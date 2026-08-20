// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
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
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto Wt = get_arg(args::Wt);
    const auto origin_w = get_arg(args::origin_w);

    DataflowBuffer dfb_one_obj(dfb::one);
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::y);

    dfb_one_obj.wait_front(onetile);  // comes from the reader

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;

    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);  // comes from the reader
    }

    constexpr bool is_zero = get_arg(args::is_zero) != 0;
    constexpr bool minus_inf = get_arg(args::minus_inf) != 0;
    using MaskOp =
        std::conditional_t<minus_inf, ckl::MaskPosInf<ckl::Dst::D0>, ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>>;
    // Compute-private intermediates (dfb::val, dfb::cal, dfb::reduce): this kernel is their only toucher, so each is
    // self-looped on the host (bound PRODUCER and CONSUMER under one accessor name).
    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const bool mask_this = do_mask_w && (col_idx == Wt - 1);
            // f(x)
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(onetile),
                ckl::CopyTile<ckl::input(dfb::x)>{},
                ckl::runtime_if(
                    mask_this,
                    ckl::CopyTile<ckl::input(dfb::mask_w, ckl::WaitPolicy::None, ckl::PopPolicy::None), ckl::Dst::D1>{},
                    MaskOp{}),
                ckl::Optional<is_zero, ckl::UnaryNe<ckl::Dst::D0>>{0u},
                ckl::Optional<!is_zero, ckl::Abs<ckl::Dst::D0>>{},
                ckl::Optional<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(dfb::val)>{});

            // calculate f(x) over dimension
            if (col_idx == 0) {
                ckl::copy<ckl::input(dfb::val), ckl::output(dfb::cal)>(ckl::IterationShape::tiles(onetile));
            } else {
                if constexpr (is_zero) {
                    ckl::add<ckl::input(dfb::val), ckl::input(dfb::cal), ckl::output(dfb::cal)>(
                        ckl::IterationShape::tiles(onetile));
                } else {
                    ckl::binary_sfpu<
                        ckl::BinaryMax<>,
                        ckl::input(dfb::val),
                        ckl::input(dfb::cal),
                        ckl::output(dfb::cal)>(ckl::IterationShape::tiles(onetile));
                }
            }
        }

        // reduce f(x)
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb::cal, dfb::one, dfb::reduce>(ckl::ReduceInputBlockShape::single());

        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::CopyTile<ckl::input(dfb::reduce)>{},
            ckl::Optional<minus_inf, ckl::Negative<ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(dfb::y)>{});
    }

    dfb_one_obj.pop_front(onetile);
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
