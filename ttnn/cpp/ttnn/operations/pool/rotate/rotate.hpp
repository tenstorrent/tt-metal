// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <string>
#include <ttnn/decorators.hpp>

namespace ttnn::operations::rotate {

struct Rotate {
    /**
     * Rotation operation with configurable interpolation.
     *
     * Rotates a tensor by an arbitrary angle around a specified center point.
     * Areas outside the rotated tensor are filled with a configurable fill value.
     *
     * Args:
     *   input_tensor: Input tensor of shape (N, H, W, C) in NHWC format
     *   angle: Rotation angle in degrees. Positive values rotate counter-clockwise
     *   center: Optional rotation center point (cx, cy). Default: tensor center
     *   fill: Fill value for areas outside the rotated tensor. Default: 0.0
     *   expand: Must be false. Only same-size rotation is supported
     *   interpolation_mode: Interpolation method - only "nearest" is supported. Default: "nearest"
     *   memory_config: Memory configuration for the output tensor
     *
     * Returns:
     *   Output tensor of shape (N, H, W, C) - same as input
     */
    static Tensor invoke(
        const Tensor& input_tensor,
        float angle,
        const std::optional<std::tuple<float, float>>& center = std::nullopt,
        float fill = 0.0f,
        bool expand = false,
        const std::string& interpolation_mode = "nearest",
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace ttnn::operations::rotate

namespace ttnn {
constexpr auto rotate = ttnn::register_operation<"ttnn::rotate", ttnn::operations::rotate::Rotate>();
}  // namespace ttnn
