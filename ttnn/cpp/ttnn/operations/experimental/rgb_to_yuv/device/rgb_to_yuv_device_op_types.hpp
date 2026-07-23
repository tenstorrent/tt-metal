// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// 3×4 coefficient matrix: each row is [weight_R, weight_G, weight_B, offset]
// for one output channel (Y, Cb, Cr).  Coefficients must map bf16 input in
// [-1, 1] directly to uint8 output in [0, 255]; no internal normalization.
//
// BT.601 example for input ∈ [-1, 1]:
//   Y  row: {32.74f,  64.28f,  12.48f, 125.5f}
//   Cb row: {-18.90f, -37.10f, 56.00f, 128.0f}
//   Cr row: {56.00f, -46.89f,  -9.11f, 128.0f}
struct YUVCoefficients {
    std::array<float, 4> y = {};   // {wy_r, wy_g, wy_b, offset_y}
    std::array<float, 4> cb = {};  // {wcb_r, wcb_g, wcb_b, offset_cb}
    std::array<float, 4> cr = {};  // {wcr_r, wcr_g, wcr_b, offset_cr}
};

// Output pixel format. Only 4:2:0 planar (ffmpeg AV_PIX_FMT_YUV420P: full-res Y
// plane, then half-res Cb and Cr planes) is implemented today; the enum is the
// extension point for adding other formats (e.g. NV12, 4:2:2) later.
enum class YUVFormat { YUV420Planar };

struct RgbToYuvParams {
    YUVCoefficients coefficients;
    YUVFormat format = YUVFormat::YUV420Planar;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct RgbToYuvInputs {
    const Tensor& input;  // CHWT bfloat16, row-major; C=3
};

}  // namespace ttnn::experimental::prim
