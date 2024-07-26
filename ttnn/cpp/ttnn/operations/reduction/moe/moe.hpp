// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"

#include "device/moe_op.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::reduction {

struct ExecuteMoe {
    static inline ttnn::Tensor operator()(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const Tensor& expert_mask_tensor,
        const Tensor& topk_mask_tensor,
        const uint16_t k,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        return operation::run(Moe{k, memory_config.value_or(input_tensor.memory_config())},
        {input_tensor, expert_mask_tensor, topk_mask_tensor},
        {},
        {optional_output_tensor},
        queue_id).at(0);
    }

    static inline auto operator()(
        const Tensor& input_tensor,
        const Tensor& expert_mask_tensor,
        const Tensor& topk_mask_tensor,
        const uint16_t k,
        const std::optional<MemoryConfig>& memory_config= std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
        constexpr uint8_t DefaultQueueId = 0;
        return operator()(DefaultQueueId, input_tensor, expert_mask_tensor, topk_mask_tensor, k, memory_config, optional_output_tensor);
    }


    static inline std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        const auto& expert_mask_tensor = input_tensors.at(1);
        const auto& topk_mask_tensor = input_tensors.at(2);
        return {Tensor(operation::get_workers_for_op_output({input_tensor, expert_mask_tensor, topk_mask_tensor}))};
    }
};

}  // namespace operations::reduction

constexpr auto moe = ttnn::register_operation_with_auto_launch_op<"ttnn::moe", ttnn::operations::reduction::ExecuteMoe>();

}  // namespace ttnn
