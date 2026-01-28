// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/experimental/where/device/where_device_operation.hpp"

#include <optional>

namespace ttnn::operations::experimental::ternary {

struct WhereOperation {
    template <ttnn::experimental::prim::FloatOrTensorConcept T, ttnn::experimental::prim::FloatOrTensorConcept U>
    static Tensor invoke(
        const Tensor& condition,
        const T& value_true,
        const U& value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        if (output_dtype.has_value() && output_tensor.has_value()) {
            TT_FATAL(
                output_dtype.value() == output_tensor.value().dtype(),
                "Both output dtype and output tensor provided dtype should match");
        }

        if constexpr (std::is_same_v<T, Tensor> and std::is_same_v<U, Tensor>) {
            auto [operation_attributes, tensor_args] = ttnn::experimental::prim::WhereDeviceOperation::invoke(
                condition, value_true, value_false, output_dtype, memory_config, std::move(output_tensor));
            return ttnn::device_operation::launch<ttnn::experimental::prim::WhereDeviceOperation>(
                operation_attributes, tensor_args);

        } else {
            TT_FATAL((!std::is_same_v<T, Tensor> || !std::is_same_v<U, Tensor>), "Scalar values are not supported!");
            return Tensor();
        }
    }
};

constexpr auto where = ttnn::register_operation<"ttnn::experimental::where", WhereOperation>();
}  // namespace ttnn::operations::experimental::ternary
