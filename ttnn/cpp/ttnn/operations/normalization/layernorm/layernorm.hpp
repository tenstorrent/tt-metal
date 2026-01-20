// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "device/layernorm_types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

// Forward declaration for parallel branch support
namespace ttnn::operations::experimental::parallel {
struct BranchDescriptor;
}

// Forward declaration for sequential step support
namespace ttnn::operations::experimental::sequential {
struct StepDescriptor;
}

namespace ttnn {
namespace operations::normalization {

struct ExecuteLayerNorm {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    // Create a branch descriptor for parallel execution
    static std::shared_ptr<ttnn::operations::experimental::parallel::BranchDescriptor> branch(
        const ttnn::Tensor& input_tensor,
        const tt::tt_metal::CoreRangeSet& cores,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const LayerNormProgramConfig>& program_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    // Create a step descriptor for sequential execution
    // Usage: auto step = ttnn::layer_norm.step(input, cores, 1e-5, weight);
    static std::shared_ptr<ttnn::operations::experimental::sequential::StepDescriptor> step(
        const ttnn::Tensor& input_tensor,
        const tt::tt_metal::CoreRangeSet& cores,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const LayerNormProgramConfig>& program_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto layer_norm =
    ttnn::register_operation<"ttnn::layer_norm", ttnn::operations::normalization::ExecuteLayerNorm>();

}  // namespace ttnn
