// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// d/dx sigmoid(x) = s(1 - s), where s = sigmoid(x).
//
// Ordering the chain as grad*s first and (1 - s) second keeps s and the running product in
// two DEST slots, so no copy between slots is needed: after MulBinary the slot holding s is
// free to be overwritten by its own complement.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"  // MulBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // Sigmoid
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"       // RsubUnary
#include "operand_reconfig.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_grad_out(dfb::grad_out);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_grad_in(dfb::grad_in);

    // 1.0f as the bit pattern the scalar SFPU ops take.
    constexpr uint32_t one_bits = 0x3f800000u;

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
        // dest[1] = input
        ckl::CopyTile<
            ckl::input(
                dfb::input,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                operand_reconfig),
            ckl::Dst::D1>{dfb_input},
        ckl::Sigmoid<ckl::Dst::D1>{},                                // dest[1] = s
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},  // dest[0] = grad_out * s
        ckl::RsubUnary<ckl::Dst::D1>{one_bits},                      // dest[1] = 1 - s
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},  // dest[0] = grad_out * s * (1 - s)
        ckl::PackTile<ckl::output(
            dfb::grad_in,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{dfb_grad_in});
}
