// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum.hpp"

#include "moreh_sum_op.hpp"

namespace ttnn::operations::moreh {

ttnn::Tensor MorehSumOperation::invoke(
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keep_batch_dim,
    const std::optional<const Tensor> output,
    const MemoryConfig &output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    return invoke(DefaultQueueId, input, dim, keep_batch_dim, output, output_mem_config, compute_kernel_config);
}

ttnn::Tensor MorehSumOperation::invoke(
    uint8_t queue_id,
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keep_batch_dim,
    const std::optional<const Tensor> output,
    const MemoryConfig &output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    return tt::operations::primary::moreh_sum(input, dim, keep_batch_dim, output, output_mem_config, compute_kernel_config, queue_id);
}

}  // namespace ttnn::operations::moreh
