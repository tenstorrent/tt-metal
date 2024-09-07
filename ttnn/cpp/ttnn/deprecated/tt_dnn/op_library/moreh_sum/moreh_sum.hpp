// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::moreh {

struct MorehSumOperation {
    static ttnn::Tensor invoke(
        const Tensor &input,
        std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
        const bool keep_batch_dim = false,
        const std::optional<const Tensor> output = std::nullopt,
        const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const Tensor &input,
        std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
        const bool keep_batch_dim = false,
        const std::optional<const Tensor> output = std::nullopt,
        const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::moreh

// TODO: remove launch_op from device operation
constexpr auto moreh_sum = ttnn::register_operation<"ttnn::moreh_sum", ttnn::operations::moreh::MorehSumOperation>();

}  // namespace ttnn
