// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "image_rotate.hpp"
#include "device/image_rotate_device_operation.hpp"

namespace ttnn::operations::image_rotate {

Tensor ImageRotate::invoke(
    const Tensor& input_tensor,
    float angle,
    const std::optional<std::tuple<float, float>>& center,
    float fill,
    bool expand,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::image_rotate(input_tensor, angle, center, fill, expand, memory_config);
}

}  // namespace ttnn::operations::image_rotate
