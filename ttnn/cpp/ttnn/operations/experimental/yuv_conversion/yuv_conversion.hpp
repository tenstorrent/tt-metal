// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/tensor/tensor.hpp"
#include "device/yuv_conversion_device_op_types.hpp"

namespace ttnn::experimental {

// Standard luma colorimetry. Each is defined by its (Kr, Kb) luma weights
// (Kg = 1 - Kr - Kb); the full YCbCr matrix is derived from these.
enum class YUVColorSpace { BT601, BT709, BT2020 };

// RGB input value range (the normalization of the input tensor's samples).
enum class RGBRange {
    MinusOneToOne,  // [-1, 1]  (e.g. VAE output)
    ZeroToOne,      // [0, 1]
};

// YUV output quantization range.
enum class YUVRange {
    Full,     // "full swing": Y,Cb,Cr in [0, 255], chroma centered at 128
    Limited,  // "studio swing": Y in [16, 235], chroma in [16, 240], centered at 128
};

// Derive the per-pixel RGB->YUV affine coefficients (the 3x4 matrix consumed by
// the op) for a standard colorspace, input range, and output range.
//
// For RGB in [lo, hi] and (Kr, Kb) with Kg = 1 - Kr - Kb, s = 1/(hi - lo):
//   Y  = yscale*s*(Kr*R + Kg*G + Kb*B) + (yoff - yscale*s*lo)
//   Cb = cscale*s/(2(1-Kb)) * (-Kr*R - Kg*G + (1-Kb)*B) + 128
//   Cr = cscale*s/(2(1-Kr)) * ((1-Kr)*R - Kg*G - Kb*B) + 128
// with (yscale, yoff, cscale) = (255, 0, 255) for Full and (219, 16, 224) for Limited.
inline prim::YUVCoefficients yuv_coefficients(YUVColorSpace color_space, RGBRange input_range, YUVRange output_range) {
    float kr = 0.f, kb = 0.f;
    switch (color_space) {
        case YUVColorSpace::BT601:
            kr = 0.299f;
            kb = 0.114f;
            break;
        case YUVColorSpace::BT709:
            kr = 0.2126f;
            kb = 0.0722f;
            break;
        case YUVColorSpace::BT2020:
            kr = 0.2627f;
            kb = 0.0593f;
            break;
    }
    const float kg = 1.f - kr - kb;

    const float lo = (input_range == RGBRange::MinusOneToOne) ? -1.f : 0.f;
    const float hi = 1.f;
    const float s = 1.f / (hi - lo);

    const bool limited = (output_range == YUVRange::Limited);
    const float yscale = limited ? 219.f : 255.f;
    const float yoff = limited ? 16.f : 0.f;
    const float cscale = limited ? 224.f : 255.f;

    const float off_y = yoff - yscale * s * lo;
    const float cbk = cscale * s / (2.f * (1.f - kb));
    const float crk = cscale * s / (2.f * (1.f - kr));

    return {
        .y = {yscale * s * kr, yscale * s * kg, yscale * s * kb, off_y},
        .cb = {-cbk * kr, -cbk * kg, cbk * (1.f - kb), 128.f},
        .cr = {crk * (1.f - kr), -crk * kg, -crk * kb, 128.f},
    };
}

// Convert a CHWT bfloat16 tensor (C=3, RGB) to YUV 4:2:0 uint8.
//
// Returns (Y, U, V) as row-major uint8 tensors:
//   Y: shape (1, H, W, T)     — full resolution luma
//   U: shape (1, H/2, W/2, T) — Cb chroma (4:2:0 subsampled)
//   V: shape (1, H/2, W/2, T) — Cr chroma (4:2:0 subsampled)
//
// The conversion coefficients are chosen by color_space / input_range /
// output_range (derived via yuv_coefficients).  Power users can instead pass an
// explicit `coefficients` matrix, which overrides the color_space/range choice.
// memory_config: output memory config; defaults to the input's. Must be
//                interleaved — sharded output is not supported.
std::tuple<Tensor, Tensor, Tensor> yuv_conversion(
    const Tensor& input,
    YUVColorSpace color_space = YUVColorSpace::BT601,
    RGBRange input_range = RGBRange::MinusOneToOne,
    YUVRange output_range = YUVRange::Limited,
    const std::optional<prim::YUVCoefficients>& coefficients = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// BT.601 / BT.709 coefficients for input ∈ [-1, 1] → limited-range uint8.
inline prim::YUVCoefficients yuv_bt601_coefficients() {
    return yuv_coefficients(YUVColorSpace::BT601, RGBRange::MinusOneToOne, YUVRange::Limited);
}
inline prim::YUVCoefficients yuv_bt709_coefficients() {
    return yuv_coefficients(YUVColorSpace::BT709, RGBRange::MinusOneToOne, YUVRange::Limited);
}

}  // namespace ttnn::experimental
