// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"

namespace ckl = compute_kernel_lib;

// GELU'(x) = 0.5*(1+tanh(z)) + 0.5*beta*x*(1+3*kappa*x^2)*(1-tanh(z)^2),
// where z = beta*(x+kappa*x^3); output = grad_out*GELU'(x).
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto dfb_grad_out_id = tt::CBIndex::c_0;
    constexpr auto dfb_input_id = tt::CBIndex::c_1;
    constexpr auto dfb_grad_in_id = tt::CBIndex::c_2;

    constexpr float kSqrt2 = 1.41421356237309504880f;          // sqrt(2)
    constexpr float kTwoOverSqrtPi = 1.12837916709551257390f;  // 2/sqrt(pi)
    constexpr float kBeta = kSqrt2 * kTwoOverSqrtPi * 0.5f;
    constexpr float kKappa = 0.044715f;

    compute_kernel_hw_startup(dfb_grad_out_id, dfb_grad_in_id);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(
                dfb_grad_out_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(dfb_input_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::CopyTile<
            ckl::input(dfb_input_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        ckl::CopyTile<
            ckl::input(dfb_input_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
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
            dfb_grad_in_id,
            ckl::ReservePolicy::PerTile,
            ckl::PushPolicy::PerTile,
            ckl::DataFormatReconfig::Disabled)>{});
}
