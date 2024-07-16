// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"

#include "ttnn/experimental/tt_dnn/op_library/run_operation.hpp"

#include "device/moe_op.hpp"
#include "ttnn/cpp/ttnn/types.hpp"

template <class Tuple,
   class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple)
{
    return std::apply([](auto&&... elems){
        return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...};
    }, std::forward<Tuple>(tuple));
}
namespace ttnn {
namespace operations::reduction {

struct ExecuteMoe {
    static inline std::vector<Tensor> execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor &input_tensor,
        const Tensor &topk_mask_tensor,
        const Tensor &expert_mask_tensor,
        const uint16_t k,
        const int8_t dim,
        const bool largest,
        const bool sorted,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors = std::nullopt) {
        return operation::run(Moe{k, dim, largest, sorted, memory_config.value_or(input_tensor.memory_config())},
        {input_tensor, topk_mask_tensor, expert_mask_tensor},
        {},
        optional_output_tensors.has_value() ? tuple_to_vector_optional(optional_output_tensors.value()) : std::vector<std::optional<Tensor>>{},
        queue_id);
    }

    static inline auto execute_on_worker_thread(
        const Tensor &input_tensor,
        const Tensor &topk_mask_tensor,
        const Tensor &expert_mask_tensor,
        const uint16_t k,
        const int8_t dim,
        const bool largest,
        const bool sorted,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<std::tuple<Tensor, Tensor>> optional_output_tensors) {
        constexpr uint8_t DefaultQueueId = 0;
        return execute_on_worker_thread(DefaultQueueId, input_tensor, topk_mask_tensor, expert_mask_tensor, k, dim, largest, sorted, memory_config, optional_output_tensors);
    }


    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        const auto& topk_mask_tensor = input_tensors.at(1);
        const auto& expert_mask_tensor = input_tensors.at(2);
        return {Tensor(operation::get_workers_for_op_output({input_tensor, topk_mask_tensor, expert_mask_tensor}))};,
                                            Tensor(operation::get_workers_for_op_output({input_tensor, topk_mask_tensor, expert_mask_tensor}))};
    }
};

}  // namespace operations::reduction

constexpr auto moe = ttnn::register_operation<ttnn::operations::reduction::ExecuteMoe>("ttnn::moe");

}  // namespace ttnn
