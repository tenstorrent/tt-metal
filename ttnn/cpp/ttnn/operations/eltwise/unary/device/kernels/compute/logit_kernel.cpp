// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"    // Log
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"  // Clamp, RsubUnary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

// Formula: logit(x) = log(x/(1-x)) -- calls clamp, rsub, div, and log tiles.

constexpr bool kDoClamp = get_arg(args::do_clamp) == 1;

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t packed_scalar1 = get_arg(args::packed_scalar1);
    const uint32_t packed_scalar2 = get_arg(args::packed_scalar2);

    // The legacy kernel boots unpack from the input and pack for the final output once;
    // tmp0 has the same element format and is only an in-kernel handoff.
    compute_kernel_hw_startup(dfb::in, dfb::out);

    // The temporary DFB holds only two tiles, and this kernel is both its producer and consumer.
    // Produce and consume one tile at a time: separating the two stages deadlocks once the producer fills the DFB.
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    dfb::in, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D0>{},
            ckl::Optional<kDoClamp, ckl::Clamp<ckl::Dst::D0>>{packed_scalar1, packed_scalar2},
            ckl::PackTile<ckl::output(
                dfb::tmp0,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Disabled)>{});

        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    dfb::tmp0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D0>{},
            ckl::CopyTile<
                ckl::input(
                    dfb::tmp0, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D1>{},
            ckl::RsubUnary<ckl::Dst::D0>{0x3F800000u},  // 1.0 - x
            ckl::DivBinary<ckl::Dst::D1, ckl::Dst::D0, ckl::Dst::D0>{},
            ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb::out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
    }
}
