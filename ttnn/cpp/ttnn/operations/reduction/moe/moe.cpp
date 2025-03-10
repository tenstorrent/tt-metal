// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/reduction/moe/moe.hpp"

#include <utility>

#include "device/moe_op.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::reduction {

ttnn::Tensor MoeOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    const uint16_t k,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return tt::tt_metal::operation::run(
               MoeDeviceOperation{k, memory_config.value_or(input_tensor.memory_config())},
               {input_tensor, expert_mask_tensor, topk_mask_tensor},
               {},
               {std::move(optional_output_tensor)},
               queue_id)
        .at(0);
}

auto MoeOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    const uint16_t k,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return invoke(
        ttnn::DefaultQueueId,
        input_tensor,
        expert_mask_tensor,
        topk_mask_tensor,
        k,
        memory_config,
        std::move(optional_output_tensor));
}

std::vector<Tensor> MoeOperation::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& input_tensor = input_tensors.at(0);
    const auto& expert_mask_tensor = input_tensors.at(1);
    const auto& topk_mask_tensor = input_tensors.at(2);
    return {Tensor(
        tt::tt_metal::operation::get_workers_for_op_output({input_tensor, expert_mask_tensor, topk_mask_tensor}))};
}

}  // namespace ttnn::operations::reduction
