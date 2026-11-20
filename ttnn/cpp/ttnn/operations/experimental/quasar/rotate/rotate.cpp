// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/experimental/quasar/rotate/rotate.hpp>
#include <ttnn/operations/experimental/quasar/rotate/device/rotate_device_operation.hpp>

namespace ttnn::operations::experimental::quasar::rotate {

Tensor rotate(
    const Tensor& input_tensor,
    float angle,
    const std::optional<std::tuple<float, float>>& center,
    float fill,
    bool expand,
    const std::string& interpolation_mode,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::qsr::rotate(input_tensor, angle, center, fill, expand, interpolation_mode, memory_config);
}

}  // namespace ttnn::operations::experimental::quasar::rotate
