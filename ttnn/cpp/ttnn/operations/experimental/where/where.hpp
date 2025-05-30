// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/common/queue_id.hpp"

#include <optional>

namespace ttnn {

namespace operations::experimental::where {

struct WhereOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& predicate,
        const Tensor& value_true,
        const Tensor& value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& predicate,
        const float value_true,
        const Tensor& value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& predicate,
        const Tensor& value_true,
        const float value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);

    static Tensor invoke(
        QueueId queue_id,
        const Tensor& predicate,
        const float value_true,
        const float value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& predicate,
        const Tensor& value_true,
        const Tensor& value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        return invoke(
            DefaultQueueId, predicate, value_true, value_false, output_dtype, memory_config, std::move(output_tensor));
    }

    static Tensor invoke(
        const Tensor& predicate,
        const float value_true,
        const Tensor& value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        return invoke(
            DefaultQueueId, predicate, value_true, value_false, output_dtype, memory_config, std::move(output_tensor));
    }

    static Tensor invoke(
        const Tensor& predicate,
        const Tensor& value_true,
        const float value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        return invoke(
            DefaultQueueId, predicate, value_true, value_false, output_dtype, memory_config, std::move(output_tensor));
    }

    static Tensor invoke(
        const Tensor& predicate,
        const float value_true,
        const float value_false,
        std::optional<const DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        return invoke(
            DefaultQueueId, predicate, value_true, value_false, output_dtype, memory_config, std::move(output_tensor));
    }
};

constexpr auto where_op = ttnn::register_operation<"ttnn::experimental::where", WhereOperation>();
}  // namespace operations::experimental::where

}  // namespace ttnn
