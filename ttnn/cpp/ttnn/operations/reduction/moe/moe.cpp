// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe.hpp"

#include "ttnn/operations/reduction/moe/device/moe_device_operation.hpp"

namespace ttnn::operations::reduction::moe {

Tensor ExecuteMoe::invoke(
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    uint16_t k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor) {
    return ttnn::prim::moe(input_tensor, expert_mask_tensor, topk_mask_tensor, k, memory_config, output_tensor);
}

}  // namespace ttnn::operations::reduction::moe
