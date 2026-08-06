// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// 3×4 coefficient matrix: each row is [weight_R, weight_G, weight_B, offset] for one output channel (Y, Cb, Cr)
struct YUVCoefficients {
    std::array<float, 4> y = {};   // {wy_r, wy_g, wy_b, offset_y}
    std::array<float, 4> cb = {};  // {wcb_r, wcb_g, wcb_b, offset_cb}
    std::array<float, 4> cr = {};  // {wcr_r, wcr_g, wcr_b, offset_cr}
};

struct YUVConversionParams {
    YUVCoefficients coefficients;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct YUVConversionInputs {
    const Tensor& input;  // CHWT bfloat16, row-major; C=3
};

}  // namespace ttnn::experimental::prim
