// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
namespace ttnn::operations::moreh::moreh_mean {
struct MorehMean {
    static Tensor invoke(
        const Tensor& input,
        const std::optional<std::variant<int64_t, std::vector<int64_t>>> dims,
        const bool keep_batch_dim,
        const std::optional<uint32_t>& divisor,
        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& output_memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_mean

namespace ttnn {
constexpr auto moreh_mean = ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_mean", ttnn::operations::moreh::moreh_mean::MorehMean>();
}
