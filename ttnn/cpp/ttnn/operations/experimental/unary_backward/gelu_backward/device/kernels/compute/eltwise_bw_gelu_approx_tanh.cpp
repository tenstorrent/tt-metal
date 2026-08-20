// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"    // FillScalar
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"         // Square, CopyDest
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // Tanh
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

#define M_SQRT2 1.41421356237309504880f    /* sqrt(2) */
#define M_2_SQRTPI 1.12837916709551257390f /* 2/sqrt(pi) */

namespace ckl = compute_kernel_lib;

// GELU'(x) = 0.5*(1+tanh(z)) + 0.5*beta*x*(1+3*kappa*x^2)*(1-tanh(z)^2),
// where z = beta*(x+kappa*x^3); output = grad_out*GELU'(x).
ALWI void gelu_tanh_fp32_chain(uint32_t num_tiles) {
    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float kKappa = 0.044715f;

    using D = ckl::Dst;
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(dfb::input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            D::D1>{},
        ckl::CopyTile<
            ckl::input(dfb::input, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            D::D2>{},
        ckl::Square<D::D1>{},
        ckl::MulBinary<D::D1, D::D2, D::D1>{},
        ckl::FillScalar<D::D3>{kKappa},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        ckl::AddBinary<D::D1, D::D2, D::D1>{},
        ckl::FillScalar<D::D3>{kBeta},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        ckl::Tanh<D::D1>{},
        ckl::CopyDest<D::D1, D::D0, DataFormat::Float32>{},
        ckl::FillScalar<D::D3>{1.0f},
        ckl::AddBinary<D::D1, D::D3, D::D1>{},
        ckl::FillScalar<D::D3>{0.5f},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        ckl::Square<D::D0>{},
        ckl::FillScalar<D::D3>{1.0f},
        ckl::SubBinary<D::D3, D::D0, D::D0>{},
        ckl::FillScalar<D::D3>{kKappa * 3.0f},
        ckl::Square<D::D2>{},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        ckl::FillScalar<D::D3>{1.0f},
        ckl::AddBinary<D::D2, D::D3, D::D2>{},
        ckl::MulBinary<D::D2, D::D0, D::D2>{},
        ckl::FillScalar<D::D3>{kBeta / 2.0f},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        ckl::CopyTile<
            ckl::input(
                dfb::grad_out, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            D::D0>{},
        ckl::CopyTile<
            ckl::input(dfb::input, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            D::D3>{},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        ckl::AddBinary<D::D1, D::D2, D::D1>{},
        ckl::MulBinary<D::D0, D::D1, D::D0>{},
        ckl::PackTile<ckl::output(
            dfb::grad_in, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}

// Keep the two slots that exceed FP32 DEST capacity template-dependent. The six-slot chain is then
// instantiated only by the non-FP32 branch below, rather than rejected while parsing an FP32 build.
template <ckl::Dst TanhSlot, ckl::Dst InputSlot>
ALWI void gelu_tanh_six_slot_chain(uint32_t num_tiles) {
    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float kKappa = 0.044715f;

    using D = ckl::Dst;
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        // grad_out -> D0 ; x -> D1 (wait owner) / D2 / InputSlot (pop owner)
        ckl::CopyTile<
            ckl::input(
                dfb::grad_out, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            D::D0>{},
        ckl::CopyTile<
            ckl::input(dfb::input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            D::D1>{},
        ckl::CopyTile<
            ckl::input(dfb::input, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            D::D2>{},
        ckl::CopyTile<
            ckl::input(dfb::input, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            InputSlot>{},
        // z = beta * (x + kappa * x^3)
        ckl::Square<D::D1>{},
        ckl::MulBinary<D::D1, D::D2, D::D1>{},
        ckl::FillScalar<D::D3>{kKappa},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        ckl::AddBinary<D::D1, D::D2, D::D1>{},
        ckl::FillScalar<D::D3>{kBeta},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        ckl::Tanh<D::D1>{},
        ckl::CopyDest<D::D1, TanhSlot>{},
        // cdf_term = 0.5 * (1 + tanh(z)) -> D1
        ckl::FillScalar<D::D3>{1.0f},
        ckl::AddBinary<D::D1, D::D3, D::D1>{},
        ckl::FillScalar<D::D3>{0.5f},
        ckl::MulBinary<D::D1, D::D3, D::D1>{},
        // TanhSlot = 1 - tanh^2
        ckl::Square<TanhSlot>{},
        ckl::FillScalar<D::D3>{1.0f},
        ckl::SubBinary<D::D3, TanhSlot, D::D3>{},
        ckl::CopyDest<D::D3, TanhSlot>{},
        // D2 = 1 + 3*kappa*x^2
        ckl::FillScalar<D::D3>{kKappa * 3.0f},
        ckl::Square<D::D2>{},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        ckl::FillScalar<D::D3>{1.0f},
        ckl::AddBinary<D::D2, D::D3, D::D2>{},
        // pdf_term = 0.5 * beta * (1 + 3*kappa*x^2) * (1 - tanh^2) -> D2
        ckl::MulBinary<D::D2, TanhSlot, D::D2>{},
        ckl::FillScalar<D::D3>{kBeta / 2.0f},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        ckl::CopyDest<InputSlot, D::D3>{},
        ckl::MulBinary<D::D2, D::D3, D::D2>{},
        // D1 = cdf_term + x * pdf_term ; D0 = grad * D1
        ckl::AddBinary<D::D1, D::D2, D::D1>{},
        ckl::MulBinary<D::D0, D::D1, D::D0>{},
        ckl::PackTile<ckl::output(
            dfb::grad_in, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}

template <bool Fp32Dest>
ALWI void gelu_tanh_chain(uint32_t num_tiles) {
    if constexpr (Fp32Dest) {
        gelu_tanh_fp32_chain(num_tiles);
    } else {
        gelu_tanh_six_slot_chain<ckl::Dst::D4, ckl::Dst::D5>(num_tiles);
    }
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    // grad_out / input are consumed from the reader; grad_in is produced for the writer.
    constexpr auto grad_format = static_cast<DataFormat>(unpack_src_format[dfb::grad_out]);
    constexpr auto input_format = static_cast<DataFormat>(unpack_src_format[dfb::input]);
    static_assert(grad_format == input_format, "GELU backward requires matching gradient and input data formats");
    static_assert(
        input_format == DataFormat::Float16_b || input_format == DataFormat::Float32,
        "GELU backward supports only bfloat16 and float32 data formats");

    compute_kernel_hw_startup(dfb::grad_out, dfb::grad_in);
    gelu_tanh_chain<ckl::get_fp32_dest_acc_enabled()>(num_tiles);
}
