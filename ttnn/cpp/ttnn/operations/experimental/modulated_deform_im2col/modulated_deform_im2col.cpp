// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modulated_deform_im2col.hpp"

#include "ttnn/operations/experimental/modulated_deform_im2col/device/modulated_deform_im2col_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor modulated_deform_im2col(
    const ttnn::Tensor& input,
    const ttnn::Tensor& offset,
    const ttnn::Tensor& mask,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::modulated_deform_im2col(
        input, offset, mask, kernel_size, stride, padding, dilation, memory_config);
}

}  // namespace ttnn::experimental
