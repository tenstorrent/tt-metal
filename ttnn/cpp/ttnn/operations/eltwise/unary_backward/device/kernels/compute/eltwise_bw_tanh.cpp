// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Uses sech^2(x) = 4*exp(-2*abs(x)) / (1+exp(-2*abs(x)))^2 to avoid catastrophic
// cancellation in the naive 1-tanh(x)^2 formula.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // TanhDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "operand_reconfig.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_grad_out(dfb::grad_out);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_grad_in(dfb::grad_in);

    compute_kernel_hw_startup(dfb::grad_out, dfb::grad_in);

    const auto shape = ckl::IterationShape::tiles(per_core_tile_cnt);

    ckl::eltwise_chain(
        shape,
        // dest[0] = grad_out
        ckl::CopyTile<
            ckl::input(
                dfb::grad_out,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                operand_reconfig),
            ckl::Dst::D0>{dfb_grad_out},
        ckl::CopyTile<
            ckl::input(
                dfb::input,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                operand_reconfig),
            ckl::Dst::D1>{dfb_input},
        ckl::TanhDerivative<ckl::Approx::Exact, ckl::Dst::D1>{},     // dest[1] = sech²(input)
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},  // dest[0] = grad_out * sech²(input)
        ckl::PackTile<ckl::output(
            dfb::grad_in,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{dfb_grad_in});
}
