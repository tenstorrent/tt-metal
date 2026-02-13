// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where.hpp"
#include <tt_stl/assert.hpp>

namespace ttnn::experimental::ternary {

// Main overload: both values are Tensors
Tensor where(
    const Tensor& condition,
    const Tensor& value_true,
    const Tensor& value_false,
    std::optional<const DataType> output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output_tensor) {
    if (output_dtype.has_value() && output_tensor.has_value()) {
        TT_FATAL(
            output_dtype.value() == output_tensor.value().dtype(),
            "Both output dtype and output tensor provided dtype should match");
    }

    auto [operation_attributes, tensor_args] = ttnn::experimental::prim::WhereDeviceOperation::invoke(
        condition, value_true, value_false, output_dtype, memory_config, std::move(output_tensor));
    return ttnn::device_operation::launch<ttnn::experimental::prim::WhereDeviceOperation>(
        operation_attributes, tensor_args);
}

// Overload: value_true is float, value_false is Tensor
Tensor where(
    const Tensor& condition [[maybe_unused]],
    float value_true [[maybe_unused]],
    const Tensor& value_false [[maybe_unused]],
    std::optional<const DataType> output_dtype [[maybe_unused]],
    const std::optional<MemoryConfig>& memory_config [[maybe_unused]],
    const std::optional<Tensor>& output_tensor [[maybe_unused]]) {
    TT_FATAL(false, "Scalar values are not supported!");
    return Tensor();
}

// Overload: value_true is Tensor, value_false is float
Tensor where(
    const Tensor& condition [[maybe_unused]],
    const Tensor& value_true [[maybe_unused]],
    float value_false [[maybe_unused]],
    std::optional<const DataType> output_dtype [[maybe_unused]],
    const std::optional<MemoryConfig>& memory_config [[maybe_unused]],
    const std::optional<Tensor>& output_tensor [[maybe_unused]]) {
    TT_FATAL(false, "Scalar values are not supported!");
    return Tensor();
}

// Overload: both values are floats
Tensor where(
    const Tensor& condition [[maybe_unused]],
    float value_true [[maybe_unused]],
    float value_false [[maybe_unused]],
    std::optional<const DataType> output_dtype [[maybe_unused]],
    const std::optional<MemoryConfig>& memory_config [[maybe_unused]],
    const std::optional<Tensor>& output_tensor [[maybe_unused]]) {
    TT_FATAL(false, "Scalar values are not supported!");
    return Tensor();
}

}  // namespace ttnn::experimental::ternary
