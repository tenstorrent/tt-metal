// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "post_combine_reduce.hpp"
#include "device/post_combine_reduce_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

ttnn::Tensor post_combine_reduce(
    const ttnn::Tensor& combine_output,
    const ttnn::Tensor& weights,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& expert_dispatch_table,
    uint32_t expert_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    // Use default L1 memory config if not specified
    auto memory_config = output_memory_config.value_or(ttnn::L1_MEMORY_CONFIG);

    return ttnn::prim::post_combine_reduce(
        combine_output, weights, indices, expert_dispatch_table, expert_dim, memory_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
