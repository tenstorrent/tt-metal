// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"    // FillScalar
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"         // Square, CopyDest
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // Tanh
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

#define M_SQRT2 1.41421356237309504880f    /* sqrt(2) */
#define M_2_SQRTPI 1.12837916709551257390f /* 2/sqrt(pi) */

namespace ckl = compute_kernel_lib;

ALWI void gelu_tanh_chain(uint32_t num_tiles) {
    constexpr auto cb_grad_out = dfb::grad_out;
    constexpr auto cb_input = dfb::input;
    constexpr auto cb_grad_in = dfb::grad_in;

    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float kKappa = 0.044715f;

    using D = ckl::Dst;
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        // grad_out -> D0 ; x -> D1 (wait owner) / D2 / D5 (pop owner)
        ckl::CopyTile<
            ckl::input(
                cb_grad_out, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            D::D0>{},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            D::D1>{},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            D::D2>{},
        ckl::CopyTile<
            ckl::input(cb_input, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            D::D5>{},
        // z = beta * (x + kappa * x^3)
        ckl::Square<D::D1>{},
        ckl::MulBinary<D::D1, D::D2, D::D1>{},
        ckl::FillScalar<D::D3>{kKappa},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        ckl::AddBinary<D::D1, D::D2, D::D1>{},
        ckl::FillScalar<D::D3>{kBeta},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        ckl::Tanh<D::D1>{},
        ckl::CopyDest<D::D1, D::D4>{},
        // cdf_term = 0.5 * (1 + tanh(z)) -> D1
        ckl::FillScalar<D::D3>{1.0f},
        ckl::AddBinary<D::D1, D::D3, D::D1>{},
        ckl::FillScalar<D::D3>{0.5f},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        // D4 = 1 - tanh^2
        ckl::Square<D::D4>{},
        ckl::FillScalar<D::D3>{1.0f},
        ckl::SubBinary<D::D3, D::D4, D::D3>{},
        ckl::CopyDest<D::D3, D::D4>{},
        // D2 = 1 + 3*kappa*x^2
        ckl::FillScalar<D::D3>{kKappa * 3.0f},
        ckl::Square<D::D2>{},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        ckl::FillScalar<D::D3>{1.0f},
        ckl::AddBinary<D::D2, D::D3, D::D2>{},
        // pdf_term = 0.5 * beta * (1 + 3*kappa*x^2) * (1 - tanh^2) -> D2
        ckl::MulBinary<D::D2, D::D4, D::D2>{},
        ckl::FillScalar<D::D3>{kBeta / 2.0f},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        ckl::CopyDest<D::D5, D::D3>{},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        // D1 = cdf_term + x * pdf_term ; D0 = grad * D1
        ckl::AddBinary<D::D1, D::D2, D::D1>{},
        ckl::MulBinary<D::D0, D::D1, D::D0>{},
        ckl::PackTile<ckl::output(
            cb_grad_in, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    compute_kernel_hw_startup(dfb::grad_out, dfb::grad_in);
    gelu_tanh_chain(num_tiles);
}
