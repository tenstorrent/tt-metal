// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/reduction/moe/moe.hpp"

#include "device/moe_op.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::reduction {

ttnn::Tensor MoeOperation::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    const uint16_t k,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return operation::run(MoeDeviceOperation{k, memory_config.value_or(input_tensor.memory_config())},
    {input_tensor, expert_mask_tensor, topk_mask_tensor},
    {},
    {optional_output_tensor},
    queue_id).at(0);
}

auto MoeOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    const uint16_t k,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    constexpr uint8_t DefaultQueueId = 0;
    return invoke(DefaultQueueId, input_tensor, expert_mask_tensor, topk_mask_tensor, k, memory_config, optional_output_tensor);
}


}  // namespace ttnn::operations::reduction
