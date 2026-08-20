// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1)

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // GeluDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

// GELU backward using the exact (non-tanh) piecewise derivative: Sollya-fitted core and corrected negative tail.
void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    // grad_out / input are consumed from the reader; grad_in is produced for the writer.
    compute_kernel_hw_startup(dfb::grad_out, dfb::grad_in);

    const auto shape = ckl::IterationShape::tiles(num_tiles);

    ckl::eltwise_chain(
        shape,
        // dest[0] = grad_out
        ckl::CopyTile<
            ckl::input(
                dfb::grad_out,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        // dest[1] = input
        ckl::CopyTile<
            ckl::input(
                dfb::input,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        // dest[1] = GELU'(input)
        ckl::GeluDerivative<ckl::Approx::Exact, ckl::Dst::D1>{},
        // dest[0] = grad_out * GELU'(input)
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb::grad_in,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}
