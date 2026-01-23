// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_operation_types.hpp"

// Forward declaration for parallel branch support
namespace ttnn::experimental::prim {
struct BranchDescriptor;
}  // namespace ttnn::experimental::prim

namespace ttnn {
namespace operations::normalization {

struct ExecuteRMSNorm {
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
    // Usage: auto branch = ttnn::rms_norm.branch(input, 1e-5, weight, cores);
    static std::shared_ptr<ttnn::experimental::prim::BranchDescriptor> branch(
        const ttnn::Tensor& input_tensor,
        const tt::tt_metal::CoreRangeSet& cores,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::normalization

constexpr auto rms_norm = ttnn::register_operation<"ttnn::rms_norm", ttnn::operations::normalization::ExecuteRMSNorm>();

}  // namespace ttnn
