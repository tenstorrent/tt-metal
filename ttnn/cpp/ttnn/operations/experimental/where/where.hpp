// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/common/queue_id.hpp"

#include "ttnn/operations/experimental/where/device/where_device_operation.hpp"

#include <optional>

namespace ttnn {

namespace operations::experimental::ternary {

struct WhereOperation {
    template <FloatOrTensorConcept T, FloatOrTensorConcept U>
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& condition,
        const T& value_true,
        const U& value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        if (output_dtype.has_value() && output_tensor.has_value()) {
            TT_FATAL(
                output_dtype.value() == output_tensor.value().get_dtype(),
                "Both output dtype and output tensor provided dtype should match");
        }

        if constexpr (std::is_same_v<T, Tensor> and std::is_same_v<U, Tensor>) {
            auto [operation_attributes, tensor_args] = WhereDeviceOperation::invoke(
                condition, value_true, value_false, output_dtype, memory_config, std::move(output_tensor));
            return ttnn::device_operation::detail::invoke<WhereDeviceOperation>(
                queue_id, operation_attributes, tensor_args);

        } else {
            TT_FATAL((!std::is_same_v<T, Tensor> || !std::is_same_v<U, Tensor>), "Scalar values are not supported!");
            return Tensor();
        }
    }
};

constexpr auto where = ttnn::register_operation<"ttnn::experimental::where", WhereOperation>();
}  // namespace operations::experimental::ternary

}  // namespace ttnn
