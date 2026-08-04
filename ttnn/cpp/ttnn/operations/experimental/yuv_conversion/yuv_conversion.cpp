// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_conversion.hpp"
#include "device/yuv_conversion_device_op.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor, Tensor> yuv_conversion(
    const Tensor& input,
    const prim::YUVCoefficients& coefficients,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    return ttnn::prim::yuv_conversion(input, coefficients, memory_config);
}

}  // namespace ttnn::experimental
