// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/rotate.hpp>
#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>

namespace ttnn::operations::rotate {

Tensor Rotate::invoke(
    const Tensor& input_tensor,
    float angle,
    const std::optional<std::tuple<float, float>>& center,
    float fill,
    bool expand,
    const std::string& interpolation_mode,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::rotate(input_tensor, angle, center, fill, expand, interpolation_mode, memory_config);
}

}  // namespace ttnn::operations::rotate
