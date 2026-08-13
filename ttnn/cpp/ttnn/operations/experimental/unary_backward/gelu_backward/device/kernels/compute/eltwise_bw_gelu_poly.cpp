// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // GeluDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    compute_kernel_hw_startup(dfb::grad_out, dfb::grad_in);

    const auto shape = ckl::IterationShape::tiles(num_tiles);

    ckl::eltwise_chain(
        shape,
        ckl::CopyTile<
            ckl::input(
                dfb::grad_out,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(
                dfb::input,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::GeluDerivative<ckl::Approx::Exact, ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb::grad_in,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}
