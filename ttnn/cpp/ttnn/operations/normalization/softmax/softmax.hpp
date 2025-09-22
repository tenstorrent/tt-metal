// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/softmax_operation_types.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <optional>

namespace ttnn::operations::normalization {
/**
 * @brief Executes the standard softmax operation on a tensor along a specified dimension.
 *
 * Computes softmax(x) = exp(x) / sum(exp(x)) along the specified dimension.
 * The operation creates a new output tensor.
 */
struct ExecuteSoftmax {
    static Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int dim,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

/**
 * @brief Executes a combined scale, mask, and softmax operation in a single fused kernel.
 *
 * Computes the following sequence of operations:
 * 1. tmp1 = scale * input_tensor (if scale is provided)
 * 2. tmp2 = tmp1 + mask (if mask is provided, broadcasts along appropriate dimensions)
 * 3. output = softmax(tmp2)
 *
 * This fused operation is commonly used in attention mechanisms where scaling and masking
 * are applied before the softmax operation. The operation creates a new output tensor.
 */
struct ExecuteScaleMaskSoftmax {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor>& mask = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        bool is_causal_mask = false,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

/**
 * @brief Executes the softmax operation in-place, modifying the input tensor directly.
 *
 * Computes softmax(x) = exp(x) / sum(exp(x)) and stores the result back into the input tensor.
 * This operation is memory-efficient as it reuses the input tensor for output, avoiding
 * additional memory allocation. Supports both default and sharded multi-core program configurations.
 */
struct ExecuteSoftmaxInPlace {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int dim = -1,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

/**
 * @brief Executes a combined scale, mask, and softmax operation in-place, modifying the input tensor directly.
 *
 * Computes the following sequence of operations in-place:
 * 1. tmp1 = scale * input_tensor (if scale is provided)
 * 2. tmp2 = tmp1 + mask (if mask is provided, with broadcasting support)
 * 3. input_tensor = softmax(tmp2)
 *
 * This fused in-place operation is commonly used in attention mechanisms and is memory-efficient
 * as it reuses the input tensor for output. The mask can be either a general attention mask or
 * a causal mask for autoregressive models.
 */
struct ExecuteScaleMaskSoftmaxInPlace {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor>& mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        bool is_causal_mask = false,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};

/**
 * @brief Specialized in-place operation for causal masked softmax with height-width dimension constraints.
 *
 * This is an experimental feature that performs scale, causal mask, and softmax operations in-place
 * with the following specific requirements and assumptions:
 * 1. Input tensor must be sharded for optimal performance
 * 2. Scale parameter must be provided (cannot be nullopt)
 * 3. Attention mask must be interleaved and have shape [1, 1, H, W] (hw_dims_only)
 * 4. Causal mask flag must be set to true
 *
 * This specialized version is optimized for transformer attention patterns where the causal mask
 * only affects the height and width dimensions, providing better performance than the general
 * scale_mask_softmax_in_place operation for these specific use cases.
 */
struct ExecuteScaleCausalMaskHWSoftmaxInPlace {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<float> scale = std::nullopt,
        const std::optional<const Tensor>& mask = std::nullopt,
        const SoftmaxProgramConfig& program_config = SoftmaxDefaultProgramConfig{},
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        bool numeric_stable = false);
};
}  // namespace ttnn::operations::normalization

namespace ttnn {

constexpr auto softmax = ttnn::register_operation<"ttnn::softmax", ttnn::operations::normalization::ExecuteSoftmax>();
constexpr auto scale_mask_softmax =
    ttnn::register_operation<"ttnn::scale_mask_softmax", ttnn::operations::normalization::ExecuteScaleMaskSoftmax>();
constexpr auto softmax_in_place =
    ttnn::register_operation<"ttnn::softmax_in_place", ttnn::operations::normalization::ExecuteSoftmaxInPlace>();
constexpr auto scale_mask_softmax_in_place = ttnn::register_operation<
    "ttnn::scale_mask_softmax_in_place",
    ttnn::operations::normalization::ExecuteScaleMaskSoftmaxInPlace>();
constexpr auto scale_causal_mask_hw_dims_softmax_in_place = ttnn::register_operation<
    "ttnn::scale_causal_mask_hw_dims_softmax_in_place",
    ttnn::operations::normalization::ExecuteScaleCausalMaskHWSoftmaxInPlace>();

}  // namespace ttnn
