// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_reduce/deepseek_moe_post_combine_reduce.hpp"
#include "ttnn/operations/experimental/deepseek_moe_post_combine_reduce/device/deepseek_moe_post_combine_reduce_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

ttnn::Tensor deepseek_moe_post_combine_reduce(
    const ttnn::Tensor& combine_output,
    const ttnn::Tensor& weights,
    uint32_t expert_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    // Use default L1 memory config if not specified
    auto memory_config = output_memory_config.value_or(ttnn::L1_MEMORY_CONFIG);

    return ttnn::prim::deepseek_moe_post_combine_reduce(
        combine_output,
        weights,
        expert_dim,
        memory_config
    );
}

}  // namespace ttnn::experimental
