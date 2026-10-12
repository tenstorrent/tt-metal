// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "shared_expert_ffn.hpp"

#include "device/shared_expert_ffn_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn {

ttnn::Tensor shared_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    RoutedExpertActivation activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const CoreCoord& core_grid_origin) {
    return ttnn::prim::shared_expert_ffn(
        x,
        gate_proj,
        up_proj,
        down_proj,
        expert_token_counts,
        global_expert_idx_table,
        expert_region_offsets,
        activation,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        core_grid_origin);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::shared_expert_ffn
