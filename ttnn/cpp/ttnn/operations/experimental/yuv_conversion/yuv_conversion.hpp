// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/tensor/tensor.hpp"
#include "device/yuv_conversion_device_op_types.hpp"

namespace ttnn::experimental {

// Convert a CHWT bfloat16 tensor (C=3, RGB, values in [-1,1]) to YUV 4:2:0 uint8
std::tuple<Tensor, Tensor, Tensor> yuv_conversion(
    const Tensor& input,
    const prim::YUVCoefficients& coefficients,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// BT.601 coefficients for input ∈ [-1, 1] bfloat16 → uint8 [0, 255].
inline prim::YUVCoefficients yuv_bt601_coefficients() {
    return {
        .y = {32.74f, 64.28f, 12.48f, 125.5f},
        .cb = {-18.90f, -37.10f, 56.00f, 128.0f},
        .cr = {56.00f, -46.89f, -9.11f, 128.0f},
    };
}

// BT.709 coefficients for input ∈ [-1, 1] bfloat16 → uint8 [0, 255].
inline prim::YUVCoefficients yuv_bt709_coefficients() {
    return {
        .y = {18.725f, 61.656f, 6.365f, 125.25f},
        .cb = {-10.257f, -33.743f, 56.00f, 128.0f},
        .cr = {56.00f, -51.499f, -6.501f, 128.0f},
    };
}

}  // namespace ttnn::experimental
