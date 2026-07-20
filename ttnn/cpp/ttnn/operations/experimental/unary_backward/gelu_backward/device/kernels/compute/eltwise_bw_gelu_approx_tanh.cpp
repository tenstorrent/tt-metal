// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"         // FillScalar
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Square, CopyDest
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Tanh
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "api/dataflow/circular_buffer.h"

#define M_SQRT2 1.41421356237309504880f    /* sqrt(2) */
#define M_2_SQRTPI 1.12837916709551257390f /* 2/sqrt(pi) */

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float kKappa = 0.044715f;

    unary_op_init_common(cb_grad_out, cb_grad_in);

    using D = ckl::Dst;
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        // grad_out -> D0 ; x -> D1 (wait owner) / D2 / D5 (pop owner)
        ckl::CopyTile<cb_grad_out, D::D0, ckl::InputLifecycle::Streaming, ckl::CopyTileReconfig::None>{},
        ckl::CopyTile<cb_input, D::D1, ckl::InputLifecycle::HeldStream, ckl::CopyTileReconfig::None>{},
        ckl::CopyTile<cb_input, D::D2, ckl::InputLifecycle::CallerManaged, ckl::CopyTileReconfig::None>{},
        ckl::CopyTile<cb_input, D::D5, ckl::InputLifecycle::NoWaitPop, ckl::CopyTileReconfig::None>{},
        // z = beta * (x + kappa * x^3)
        ckl::Square<D::D1>{},                   // D1 = x^2
        ckl::MulBinary<D::D1, D::D2, D::D1>{},  // D1 = x^3
        ckl::FillScalar<D::D3>{kKappa},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},  // D1 = kappa*x^3
        ckl::AddBinary<D::D1, D::D2, D::D1>{},  // D1 = x + kappa*x^3
        ckl::FillScalar<D::D3>{kBeta},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},  // D1 = z
        ckl::Tanh<D::D1>{},                     // D1 = tanh(z)
        ckl::CopyDest<D::D1, D::D4>{},          // D4 = tanh(z)
        // cdf_term = 0.5 * (1 + tanh(z))  -> D1
        ckl::FillScalar<D::D3>{1.0f},
        ckl::AddBinary<D::D1, D::D3, D::D1>{},
        ckl::FillScalar<D::D3>{0.5f},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        // D4 = 1 - tanh^2
        ckl::Square<D::D4>{},
        ckl::FillScalar<D::D3>{1.0f},
        ckl::SubBinary<D::D3, D::D4, D::D3>{},  // D3 = 1 - tanh^2
        ckl::CopyDest<D::D3, D::D4>{},          // D4 = 1 - tanh^2
        // D2 = (1 + 3*kappa*x^2)
        ckl::FillScalar<D::D3>{kKappa * 3.0f},
        ckl::Square<D::D2>{},                   // D2 = x^2
        ckl::MulBinary<D::D2, D::D3, D::D2>{},  // D2 = 3*kappa*x^2
        ckl::FillScalar<D::D3>{1.0f},
        ckl::AddBinary<D::D2, D::D3, D::D2>{},  // D2 = 1 + 3*kappa*x^2
        // pdf_term = 0.5 * beta * (1 + 3*kappa*x^2) * (1 - tanh^2)  -> D2
        ckl::MulBinary<D::D2, D::D4, D::D2>{},
        ckl::FillScalar<D::D3>{kBeta / 2.0f},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        // D2 = x * pdf_term
        ckl::CopyDest<D::D5, D::D3>{},  // D3 = x
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        // D1 = cdf_term + x * pdf_term ; D0 = grad * D1
        ckl::AddBinary<D::D1, D::D2, D::D1>{},
        ckl::MulBinary<D::D0, D::D1, D::D0>{},
        ckl::PackTile<cb_grad_in, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}
