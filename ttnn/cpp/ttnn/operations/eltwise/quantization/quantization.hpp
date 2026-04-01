// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

Tensor quantize(
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& scale,
    const std::variant<Tensor, int32_t>& zero_point,
    std::optional<int32_t> axis = std::nullopt,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

Tensor requantize(
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& in_scale,
    const std::variant<Tensor, int32_t>& in_zero_point,
    const std::variant<Tensor, float>& out_scale,
    const std::variant<Tensor, int32_t>& out_zero_point,
    std::optional<int32_t> axis = std::nullopt,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

Tensor dequantize(
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& scale,
    const std::variant<Tensor, int32_t>& zero_point,
    std::optional<int32_t> axis = std::nullopt,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt);

}  // namespace ttnn
