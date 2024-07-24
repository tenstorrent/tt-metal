// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <functional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {

namespace operations {

namespace ternary {

namespace constants {
   const uint8_t DefaultQueueId = 0;
}

struct WhereOp {
  static Tensor _where(uint8_t queue_id, const Tensor& predicate, const Tensor& value_true, const Tensor& value_false, const std::optional<MemoryConfig>& output_mem_config = std::nullopt, std::optional<Tensor> output_tensor = std::nullopt);
  static Tensor _where(uint8_t queue_id, const Tensor& predicate, const float value_true, const Tensor& value_false, const std::optional<MemoryConfig>& output_mem_config = std::nullopt, std::optional<Tensor> output_tensor = std::nullopt);
  static Tensor _where(uint8_t queue_id, const Tensor& predicate, const Tensor& value_true, const float value_false, const std::optional<MemoryConfig>& output_mem_config = std::nullopt, std::optional<Tensor> output_tensor = std::nullopt);
  static Tensor _where(uint8_t queue_id, const Tensor& predicate, const float value_true, const float value_false, const std::optional<MemoryConfig>& output_mem_config = std::nullopt, std::optional<Tensor> output_tensor = std::nullopt);
};

struct ExecuteTernaryWhere
{
    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& predicate,
        const Tensor& value_true,
        const Tensor& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt){
            return WhereOp::_where(queue_id, predicate, value_true, value_false, memory_config.value_or(predicate.memory_config()), output_tensor);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& predicate,
        const float value_true,
        const Tensor& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt){
            return WhereOp::_where(queue_id, predicate, value_true, value_false, memory_config.value_or(predicate.memory_config()), output_tensor);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& predicate,
        const Tensor& value_true,
        const float value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt){
            return WhereOp::_where(queue_id, predicate, value_true, value_false, memory_config.value_or(predicate.memory_config()), output_tensor);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& predicate,
        const float value_true,
        const float value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt){
            return WhereOp::_where(queue_id, predicate, value_true, value_false, memory_config.value_or(predicate.memory_config()), output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& predicate,
        const Tensor& value_true,
        const Tensor& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt){
            return WhereOp::_where(constants::DefaultQueueId, predicate, value_true, value_false, memory_config.value_or(predicate.memory_config()), output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& predicate,
        const float value_true,
        const Tensor& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt){
            return WhereOp::_where(constants::DefaultQueueId, predicate, value_true, value_false, memory_config.value_or(predicate.memory_config()), output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& predicate,
        const Tensor& value_true,
        const float value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt){
            return WhereOp::_where(constants::DefaultQueueId, predicate, value_true, value_false, memory_config.value_or(predicate.memory_config()), output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& predicate,
        const float value_true,
        const float value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt){
            return WhereOp::_where(constants::DefaultQueueId, predicate, value_true, value_false, memory_config.value_or(predicate.memory_config()), output_tensor);
    }

};

}  // namespace ternary
}  // namespace operations

constexpr auto where = ttnn::register_operation<"ttnn::where", operations::ternary::ExecuteTernaryWhere>();

}  // namespace ttnn
