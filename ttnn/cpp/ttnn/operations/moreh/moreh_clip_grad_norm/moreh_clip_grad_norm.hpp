// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm

namespace ttnn {

/**
 * @brief Clips gradient norm of an iterable of tensors.
 *
 * This operation computes the total norm of gradients across all input tensors and clips
 * them to a maximum value. It is commonly used in training to prevent exploding gradients.
 * The operation modifies the input tensors in-place.
 *
 * The total norm is computed as:
 *   total_norm = (sum(|grad_i|^norm_type for all grads))^(1/norm_type)
 *
 * If total_norm > max_norm, each gradient is scaled by:
 *   clip_coef = max_norm / (total_norm + 1e-6)
 *
 * Args:
 *     inputs (std::vector<Tensor>): Input tensors containing gradients to be clipped
 *     max_norm (float): Maximum norm value
 *     norm_type (float): Type of the norm (e.g., 2.0 for L2 norm)
 *     error_if_nonfinite (bool): If true, throws error when total norm is non-finite
 *
 * Keyword Args:
 *     total_norm (std::optional<const Tensor>): Optional pre-allocated output tensor for total norm
 *     memory_config (std::optional<MemoryConfig>): Memory configuration for intermediate tensors
 *     compute_kernel_config (std::optional<DeviceComputeKernelConfig>): Compute kernel configuration
 *
 * Returns:
 *     Tensor: Total norm of the gradients before clipping
 */
Tensor moreh_clip_grad_norm(
    const std::vector<Tensor>& inputs,
    float max_norm,
    float norm_type = 2.0f,
    bool error_if_nonfinite = false,
    const std::optional<const Tensor>& total_norm = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn
