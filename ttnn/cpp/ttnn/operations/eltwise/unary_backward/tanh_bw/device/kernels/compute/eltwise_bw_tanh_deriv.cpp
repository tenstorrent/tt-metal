// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Avoids catastrophic cancellation in the naive 1 - tanh²(x) formula.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // TanhDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto dfb_grad_out_id = tt::CBIndex::c_0;
    constexpr auto dfb_input_id = tt::CBIndex::c_1;
    constexpr auto dfb_grad_in_id = tt::CBIndex::c_2;

    compute_kernel_hw_startup(dfb_grad_out_id, dfb_grad_in_id);

    const auto shape = ckl::EltwiseShape::tiles(per_core_tile_cnt);

    ckl::eltwise_chain(
        shape,
        ckl::CopyTile<
            ckl::input(
                dfb_grad_out_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(
                dfb_input_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::TanhDerivative<ckl::Approx::Exact, ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_grad_in_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}
