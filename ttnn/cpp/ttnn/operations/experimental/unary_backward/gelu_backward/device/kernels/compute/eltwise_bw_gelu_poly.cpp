// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // GeluDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr auto cb_grad_out = dfb::grad_out;
    constexpr auto cb_input = dfb::input;
    constexpr auto cb_grad_in = dfb::grad_in;

    compute_kernel_hw_startup(cb_grad_out, cb_grad_in);

    const auto shape = ckl::EltwiseShape::tiles(num_tiles);

    ckl::eltwise_chain(
        shape,
        ckl::CopyTile<
            ckl::input(
                cb_grad_out,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(
                cb_input,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::GeluDerivative<ckl::Approx::Exact, ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            cb_grad_in,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}
