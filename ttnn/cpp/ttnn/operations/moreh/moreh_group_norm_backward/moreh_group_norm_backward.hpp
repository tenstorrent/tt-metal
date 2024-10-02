// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
struct MorehGroupNormBackward {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& mean,
        const Tensor& rstd,
        const uint32_t num_groups,
        const std::vector<bool>& are_required_outputs,
        const std::optional<const Tensor> gamma,
        const std::optional<const Tensor> input_grad,
        const std::optional<const Tensor> gamma_grad,
        const std::optional<const Tensor> beta_grad,
        const std::optional<MemoryConfig>& input_grad_memory_config,
        const std::optional<MemoryConfig>& gamma_grad_memory_config,
        const std::optional<MemoryConfig>& beta_grad_memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward

namespace ttnn {
constexpr auto moreh_group_norm_backward = ttnn::register_operation<
    "ttnn::moreh_group_norm_backward",
    ttnn::operations::moreh::moreh_group_norm_backward::MorehGroupNormBackward>();
}
