// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    constexpr float kSqrt2 = 1.41421356237309504880f;          // sqrt(2)
    constexpr float kTwoOverSqrtPi = 1.12837916709551257390f;  // 2/sqrt(pi)
    constexpr float kBeta = kSqrt2 * kTwoOverSqrtPi * 0.5f;
    constexpr float kKappa = 0.044715f;

    compute_kernel_hw_startup(cb_grad_out, cb_grad_in);

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(
                cb_grad_out, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D5>{},
        ckl::Square<ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::FillScalar<ckl::Dst::D3>{kKappa},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D3, ckl::Dst::D1>{},
        ckl::AddBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::FillScalar<ckl::Dst::D3>{kBeta},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D3, ckl::Dst::D1>{},
        ckl::Tanh<ckl::Dst::D1>{},
        ckl::CopyDest<ckl::Dst::D1, ckl::Dst::D4>{},
        ckl::FillScalar<ckl::Dst::D3>{1.0f},
        ckl::AddBinary<ckl::Dst::D1, ckl::Dst::D3, ckl::Dst::D1>{},
        ckl::FillScalar<ckl::Dst::D3>{0.5f},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D3, ckl::Dst::D1>{},
        ckl::Square<ckl::Dst::D4>{},
        ckl::FillScalar<ckl::Dst::D3>{1.0f},
        ckl::SubBinary<ckl::Dst::D3, ckl::Dst::D4, ckl::Dst::D3>{},
        ckl::CopyDest<ckl::Dst::D3, ckl::Dst::D4>{},
        ckl::FillScalar<ckl::Dst::D3>{kKappa * 3.0f},
        ckl::Square<ckl::Dst::D2>{},
        ckl::MulBinary<ckl::Dst::D2, ckl::Dst::D3, ckl::Dst::D2>{},
        ckl::FillScalar<ckl::Dst::D3>{1.0f},
        ckl::AddBinary<ckl::Dst::D2, ckl::Dst::D3, ckl::Dst::D2>{},
        ckl::MulBinary<ckl::Dst::D2, ckl::Dst::D4, ckl::Dst::D2>{},
        ckl::FillScalar<ckl::Dst::D3>{kBeta / 2.0f},
        ckl::MulBinary<ckl::Dst::D2, ckl::Dst::D3, ckl::Dst::D2>{},
        ckl::CopyDest<ckl::Dst::D5, ckl::Dst::D3>{},
        ckl::MulBinary<ckl::Dst::D2, ckl::Dst::D3, ckl::Dst::D2>{},
        ckl::AddBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            cb_grad_in, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
