// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "where.hpp"

#include <functional>
#include <type_traits>
#include <utility>
#include <variant>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/experimental/where/device/where_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations::experimental::where {

// where - ternary operator y = (predicate) ? value_true : value_false; elementwise
// y = (predicate >= 0)*value_true + (predicate < 0)*value_false

namespace details {
using FloatOrTensor = std::variant<Tensor, float>;

template <FloatOrTensorConcept T, FloatOrTensorConcept U>
Tensor where_impl(
    QueueId queue_id,
    const Tensor& predicate,
    const T& value_true,
    const U& value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    // TODO: missing const dtype
    auto dtype = ttnn::DataType::BFLOAT16;
    if constexpr (std::is_same_v<T, Tensor> and std::is_same_v<U, Tensor>) {
        return ttnn::prim::where_impl(
            queue_id, predicate, value_true, value_false, dtype, output_mem_config, std::move(output_tensor));
    } else {
        TT_FATAL((!std::is_same_v<T, Tensor> || !std::is_same_v<U, Tensor>), "Scalar values are not supported!");
        return Tensor();
    }
}
}  // namespace details

}  // namespace operations::experimental::where
Tensor operations::experimental::where::WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const Tensor& value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return details::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor operations::experimental::where::WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const float value_true,
    const Tensor& value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return details::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor operations::experimental::where::WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const Tensor& value_true,
    const float value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return details::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

Tensor operations::experimental::where::WhereOperation::invoke(
    QueueId queue_id,
    const Tensor& predicate,
    const float value_true,
    const float value_false,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return details::where_impl(
        queue_id,
        predicate,
        value_true,
        value_false,
        output_mem_config.value_or(predicate.memory_config()),
        std::move(output_tensor));
}

}  // namespace ttnn
