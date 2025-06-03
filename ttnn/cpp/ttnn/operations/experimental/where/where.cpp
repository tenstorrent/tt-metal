// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "where.hpp"

#include <type_traits>
#include <utility>

#include "ttnn/operations/experimental/where/device/where_device_operation.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {
namespace operations::ternary::experimental {

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
// y = (predicate >= 0)*value_true + (predicate < 0)*value_false

namespace details {

template <FloatOrTensorConcept T, FloatOrTensorConcept U>
Tensor where_impl(
    QueueId queue_id,
    const Tensor& predicate,
    const T& value_true,
    const U& value_false,
    std::optional<const DataType> output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    if (output_dtype.has_value() && output_tensor.has_value()) {
        TT_FATAL(
            output_dtype.value() == output_tensor.value().get_dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }

    if constexpr (std::is_same_v<T, Tensor> and std::is_same_v<U, Tensor>) {
        // TODO: no need to have invoke name anymore
        auto [operation_attributes, tensor_args] = WhereDeviceOperation::invoke(
            predicate, value_true, value_false, output_dtype, output_mem_config, std::move(output_tensor));
        return ttnn::device_operation::detail::invoke<WhereDeviceOperation>(
            queue_id, operation_attributes, tensor_args);

    } else {
        TT_FATAL((!std::is_same_v<T, Tensor> || !std::is_same_v<U, Tensor>), "Scalar values are not supported!");
        return Tensor();
    }
}
}  // namespace details

}  // namespace operations::ternary::experimental
Tensor operations::ternary::experimental::WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    std::optional<const DataType> output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return details::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_dtype,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor operations::ternary::experimental::WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const float value_true,
    const Tensor& value_false,
    std::optional<const DataType> output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return details::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_dtype,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor operations::ternary::experimental::WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const float value_false,
    std::optional<const DataType> output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return details::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_dtype,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor operations::ternary::experimental::WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const float value_true,
    const float value_false,
    std::optional<const DataType> output_dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return details::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_dtype,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

}  // namespace ttnn
