// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

// Shared helpers for rms_norm_{pre,post}_all_gather_bw.
// For y = gamma * x / rms with rms = sqrt(E[x^2] + eps):
//   dL/dx = g - x * E[x * g] / rms^2,  g = gamma * dL/dy / rms
//   dL/dgamma = SUM_rows(dL/dy * x / rms)
// Only E[x^2] and E[x * g] are all-gathered; everything else is local to the shard.
namespace ttnn::operations::normalization::rmsnorm_distributed_bw {

DeviceComputeKernelConfig resolve_compute_kernel_config(
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config, const Tensor& input);

void validate_bw_inputs(
    const Tensor& input,
    const Tensor& output_grad,
    const std::optional<const Tensor>& weight,
    std::string_view op_name);

void validate_stats_tensor(
    const Tensor& stats, const Tensor& input, std::string_view tensor_name, std::string_view op_name);

uint32_t num_devices_in_stats(const Tensor& stats);

// Stats are one tile column per device (SUM in column 0); sums the last dim and divides by full row width.
Tensor mean_from_gathered_stats(
    const Tensor& stats, uint32_t local_width, const DeviceComputeKernelConfig& compute_kernel_config);

Tensor rms_from_gathered_stats(
    const Tensor& stats, uint32_t local_width, float epsilon, const DeviceComputeKernelConfig& compute_kernel_config);

Tensor x_times_gained(
    const Tensor& input, const Tensor& output_grad, const Tensor& rms, const std::optional<const Tensor>& weight);

// Pad to one tile column, value in column 0, matching rms_norm_pre_all_gather.
Tensor to_stats_layout(const Tensor& tensor);

// Returns {input_grad, weight_grad}; weight_grad is nullopt when weight is unset.
// `scale` is E[x * g] over the full row.
std::vector<std::optional<Tensor>> apply_backward(
    const Tensor& input,
    const Tensor& output_grad,
    const Tensor& rms,
    const Tensor& scale,
    const std::optional<const Tensor>& weight,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::operations::normalization::rmsnorm_distributed_bw
