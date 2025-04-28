// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"

namespace ttnn {
namespace operations::reduction {

struct MoeOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        const Tensor& expert_mask_tensor,
        const Tensor& topk_mask_tensor,
        const uint16_t k,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static auto invoke(
        const Tensor& input_tensor,
        const Tensor& expert_mask_tensor,
        const Tensor& topk_mask_tensor,
        const uint16_t k,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};
}  // namespace operations::reduction

constexpr auto moe = ttnn::register_operation<"ttnn::moe", ttnn::operations::reduction::MoeOperation>();

}  // namespace ttnn
