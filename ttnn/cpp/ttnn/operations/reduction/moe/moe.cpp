// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe.hpp"

#include "ttnn/operations/reduction/moe/device/moe_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::reduction::moe {

Tensor moe(
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    uint16_t k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& preallocated_output_tensor) {
    using OperationType = MoeDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .k = k, .output_memory_config = memory_config.value_or(tt::tt_metal::MemoryConfig{})},
        OperationType::tensor_args_t{
            .input = input_tensor,
            .expert_mask = expert_mask_tensor,
            .topk_mask = topk_mask_tensor,
            .preallocated_output = preallocated_output_tensor});
}

}  // namespace ttnn::operations::reduction::moe
