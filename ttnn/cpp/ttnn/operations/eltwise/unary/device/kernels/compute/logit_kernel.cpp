// SPDX-FileCopyrightText: ? 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"          // Log1p
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"          // Abs, Sign
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"        // Clamp, RsubUnary, SubUnary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"   // AddBinary, DivBinary, MulBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"  // BinaryMin
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"       // Optional
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

// Formula: logit(x) = sign(x - 0.5) * log1p(2 * |x - 0.5| / min(x, 1 - x))
// Evaluating log1p on the non-negative argument 2 * |x - 0.5| / min(x, 1 - x)
// avoids catastrophic cancellation near x = 0.5 (where x / (1 - x) -> 1) and
// near x -> 0 (where (2x - 1) / (1 - x) -> -1), while preserving -inf at x = 0,
// +inf at x = 1, and NaN outside [0, 1] (where min(x, 1 - x) < 0 yields a < -2).

constexpr bool kDoClamp = get_arg(args::do_clamp) == 1;

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t packed_scalar1 = get_arg(args::packed_scalar1);
    const uint32_t packed_scalar2 = get_arg(args::packed_scalar2);

    DataflowBuffer dfb_in_obj(dfb::in);
    DataflowBuffer dfb_tmp0_obj(dfb::tmp0);
    DataflowBuffer dfb_out_obj(dfb::out);

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
                ckl::Dst::D0>{dfb_in_obj},
            ckl::Optional<kDoClamp, ckl::Clamp<ckl::Dst::D0>>{packed_scalar1, packed_scalar2},
            ckl::PackTile<ckl::output(
                dfb::tmp0, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{
                dfb_tmp0_obj});

        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    dfb::tmp0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D0>{dfb_tmp0_obj},
            ckl::CopyTile<
                ckl::input(
                    dfb::tmp0, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D1>{dfb_tmp0_obj},
            ckl::CopyTile<
                ckl::input(
                    dfb::tmp0, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D2>{dfb_tmp0_obj},
            ckl::RsubUnary<ckl::Dst::D0>{0x3F800000u},                   // D0 = 1.0 - x
            ckl::BinaryMin<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},  // D0 = min(1.0 - x, x)
            ckl::SubUnary<ckl::Dst::D1>{0x3F000000u},                    // D1 = x - 0.5
            ckl::SubUnary<ckl::Dst::D2>{0x3F000000u},                    // D2 = x - 0.5
            ckl::Sign<ckl::Dst::D2>{},                                   // D2 = sign(x - 0.5)
            ckl::Abs<ckl::Dst::D1>{},                                    // D1 = |x - 0.5|
            ckl::AddBinary<ckl::Dst::D1, ckl::Dst::D1, ckl::Dst::D1>{},  // D1 = 2 * |x - 0.5|
            ckl::DivBinary<ckl::Dst::D1, ckl::Dst::D0, ckl::Dst::D0>{},  // D0 = 2 * |x - 0.5| / min(x, 1 - x)
            ckl::Log1p<ckl::Approx::Exact, ckl::Dst::D0>{},              // D0 = log1p(...)
            ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D0>{},  // D0 = sign(x - 0.5) * log1p(...)
            ckl::PackTile<ckl::output(
                dfb::out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{
                dfb_out_obj});
    }
}
