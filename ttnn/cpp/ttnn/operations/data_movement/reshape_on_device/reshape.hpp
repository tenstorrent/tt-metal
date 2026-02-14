// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

// TODO: unify with ttnn::reshape in core.cpp
ttnn::Tensor reshape_on_device(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& logical_output_shape,
    const ttnn::Shape& padded_output_shape,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

ttnn::Tensor reshape_on_device(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& logical_output_shape,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

ttnn::Tensor reshape_on_device(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

// Python binding overload: takes W, Z, Y, X as separate parameters
ttnn::Tensor reshape_on_device(
    const ttnn::Tensor& input_tensor,
    int W,
    int Z,
    int Y,
    int X,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

}  // namespace ttnn
