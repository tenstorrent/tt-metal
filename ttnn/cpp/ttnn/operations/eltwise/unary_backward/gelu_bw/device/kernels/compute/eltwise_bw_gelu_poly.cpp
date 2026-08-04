// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // GeluDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    compute_kernel_hw_startup(cb_grad_out, cb_grad_in);

    const auto shape = ckl::EltwiseShape::tiles(per_core_tile_cnt);

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
