// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
struct MorehGroupNorm {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& input,
        const uint32_t num_groups,
        const float eps,
        const std::optional<const Tensor> gamma,
        const std::optional<const Tensor> beta,
        const std::vector<bool>& are_required_outputs,
        const std::optional<const Tensor> output,
        const std::optional<const Tensor> mean,
        const std::optional<const Tensor> rstd,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<MemoryConfig>& mean_memory_config,
        const std::optional<MemoryConfig>& rstd_memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_group_norm

namespace ttnn {
constexpr auto moreh_group_norm =
    ttnn::register_operation<"ttnn::moreh_group_norm", ttnn::operations::moreh::moreh_group_norm::MorehGroupNorm>();
}
