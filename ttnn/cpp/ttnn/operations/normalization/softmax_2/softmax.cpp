// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.hpp"

namespace ttnn::operations::normalization {
Tensor ExecuteSoftmax::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    int dim,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // TODO: Fill
    return Tensor{};  // TODO: Fix
}

Tensor ExecuteScaleMaskSoftmax::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<float> scalet,
    const std::optional<const Tensor>& mask,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // TODO: Fill
    return Tensor{};  // TODO: Fix
}

Tensor ExecuteSoftmaxInPlace::invoke(
    const ttnn::Tensor& input_tensor,
    const SoftmaxProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // TODO: Fill
    return Tensor{};  // TODO: Fix
}

Tensor ExecuteScaleMaskSoftmaxInPlace::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const SoftmaxProgramConfig& program_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // TODO: Fill
    return Tensor{};  // TODO: Fix
}

Tensor ExecuteScaleCausalMaskHWSoftmaxInPlace::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const SoftmaxProgramConfig& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // TODO: Fill
    return Tensor{};  // TODO: Fix
}
}  // namespace ttnn::operations::normalization
