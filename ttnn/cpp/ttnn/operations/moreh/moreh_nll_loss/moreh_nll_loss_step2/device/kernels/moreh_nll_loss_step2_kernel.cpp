// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr auto per_core_tile_cnt = get_arg(args::per_core_tile_cnt);
    using D = ckl::Dst;

#if defined(WEIGHT)
    constexpr bool has_weight = true;
#else
    constexpr bool has_weight = false;
#endif

#if defined(DIVISOR)
    constexpr bool has_divisor = true;
#else
    constexpr bool has_divisor = false;
#endif

    compute_kernel_hw_startup(dfb::tmp_weight, dfb::tmp_input, dfb::output);

    // `dfb::divisor` is not declared at all in the sum-reduction program.  This
    // must be a preprocessor guard rather than `if constexpr`: non-dependent
    // DFB token names are resolved before the discarded branch is eliminated.
#if defined(DIVISOR)
    {
        ckl::unary<
            ckl::Recip<D::D0>,
            ckl::input(
                dfb::divisor, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckernel::moreh_data_format_reconfig),
            ckl::output(
                dfb::divisor_recip,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>(ckl::IterationShape::one_tile());
    }
#endif

    // Keep the current algorithm's operation order: negate, then apply the optional weight,
    // then the scalar reciprocal. Re-associating these products changes low-precision rounding.
    if constexpr (has_weight || has_divisor) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(per_core_tile_cnt),
            ckl::CopyTile<ckl::input(
                dfb::tmp_input,
                ckl::WaitPolicy::PerTile,
                ckl::PopPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<ckl::output(
                dfb::tmp1,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{});
    }

    if constexpr (has_weight) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(per_core_tile_cnt),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    dfb::tmp1, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckernel::moreh_data_format_reconfig),
                ckl::input(
                    dfb::tmp_weight,
                    ckl::BroadcastDim::None,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    ckl::InputTileMapping::Scalar,
                    ckernel::moreh_data_format_reconfig)>{},
            ckl::PackTile<ckl::output(
                has_divisor ? dfb::tmp3 : dfb::output,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{});
    }

    if constexpr (has_divisor) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(per_core_tile_cnt),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    has_weight ? dfb::tmp3 : dfb::tmp1,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    ckernel::moreh_data_format_reconfig),
                ckl::input(
                    dfb::divisor_recip,
                    ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Scalar,
                    ckernel::moreh_data_format_reconfig)>{},
            ckl::PackTile<ckl::output(
                dfb::output,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{});
    }

    if constexpr (!has_weight && !has_divisor) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(per_core_tile_cnt),
            ckl::CopyTile<ckl::input(
                dfb::tmp_input,
                ckl::WaitPolicy::PerTile,
                ckl::PopPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<ckl::output(
                dfb::output,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{});
    }
}
