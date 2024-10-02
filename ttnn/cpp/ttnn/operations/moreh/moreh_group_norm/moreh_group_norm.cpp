// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm.hpp"

#include "device/moreh_group_norm_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
std::vector<std::optional<Tensor>> MorehGroupNorm::invoke(
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
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_group_norm(
        input,
        num_groups,
        eps,
        gamma,
        beta,
        are_required_outputs,
        output,
        mean,
        rstd,
        memory_config,
        mean_memory_config,
        rstd_memory_config,
        compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_group_norm
