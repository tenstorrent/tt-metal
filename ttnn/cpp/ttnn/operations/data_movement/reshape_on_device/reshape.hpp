// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

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
    ttsl::Span<const int32_t> shape_vector,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

}  // namespace ttnn
