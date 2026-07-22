// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_conversion.hpp"
#include "device/yuv_conversion_device_op.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor, Tensor> yuv_conversion(
    const Tensor& input,
    YUVColorSpace color_space,
    RGBRange input_range,
    YUVRange output_range,
    const std::optional<prim::YUVCoefficients>& coefficients,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    const prim::YUVCoefficients coeffs =
        coefficients.value_or(yuv_coefficients(color_space, input_range, output_range));
    return ttnn::prim::yuv_conversion(input, coeffs, memory_config);
}

}  // namespace ttnn::experimental
