// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <string>
#include <ttnn/decorators.hpp>

namespace ttnn::operations::rotate {

Tensor rotate(
    const Tensor& input_tensor,
    float angle,
    const std::optional<std::tuple<float, float>>& center = std::nullopt,
    float fill = 0.0f,
    bool expand = false,
    const std::string& interpolation_mode = "nearest",
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::rotate

namespace ttnn {
using ttnn::operations::rotate::rotate;
}  // namespace ttnn
