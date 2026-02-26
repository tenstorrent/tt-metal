// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/where/device/where_device_operation.hpp"

#include <optional>

namespace ttnn::experimental::ternary {

// Main overload: both values are Tensors
Tensor where(
    const Tensor& condition,
    const Tensor& value_true,
    const Tensor& value_false,
    std::optional<const DataType> output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> output_tensor = std::nullopt);

// Overload: value_true is float, value_false is Tensor
Tensor where(
    const Tensor& condition,
    float value_true,
    const Tensor& value_false,
    std::optional<const DataType> output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

// Overload: value_true is Tensor, value_false is float
Tensor where(
    const Tensor& condition,
    const Tensor& value_true,
    float value_false,
    std::optional<const DataType> output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

// Overload: both values are floats
Tensor where(
    const Tensor& condition,
    float value_true,
    float value_false,
    std::optional<const DataType> output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn::experimental::ternary
