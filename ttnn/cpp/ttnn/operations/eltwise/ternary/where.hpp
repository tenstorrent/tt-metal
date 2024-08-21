// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn {

namespace operations {

namespace ternary {

struct WhereOperation
{

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& predicate,
        const Tensor& value_true,
        const Tensor& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& predicate,
        const float value_true,
        const Tensor& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& predicate,
        const Tensor& value_true,
        const float value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& predicate,
        const float value_true,
        const float value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& predicate,
        const Tensor& value_true,
        const Tensor& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        return invoke (
            DefaultQueueId,
            predicate,
            value_true,
            value_false,
            memory_config,
            output_tensor);
    }

    static Tensor invoke(
        const Tensor& predicate,
        const float value_true,
        const Tensor& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        return invoke (
            DefaultQueueId,
            predicate,
            value_true,
            value_false,
            memory_config,
            output_tensor);
    }

    static Tensor invoke(
        const Tensor& predicate,
        const Tensor& value_true,
        const float value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        return invoke (
            DefaultQueueId,
            predicate,
            value_true,
            value_false,
            memory_config,
            output_tensor);
    }

    static Tensor invoke(
        const Tensor& predicate,
        const float value_true,
        const float value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt) {
        return invoke (
            DefaultQueueId,
            predicate,
            value_true,
            value_false,
            memory_config,
            output_tensor);
    }
};

}  // namespace ternary
}  // namespace operations

constexpr auto where = ttnn::register_operation_with_auto_launch_op<"ttnn::where", operations::ternary::WhereOperation>();

}  // namespace ttnn
