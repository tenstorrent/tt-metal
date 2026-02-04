// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::operations::experimental::minimal_matmul {

// Re-export the config type for backward compatibility
using MinimalMatmulConfig = ttnn::experimental::prim::MinimalMatmulConfig;

struct ExecuteMinimalMatmul {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weight_tensor,
        const std::optional<ttnn::Tensor>& bias_tensor,
        std::optional<unary::UnaryWithParam> fused_activation,
        const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<const DataType> dtype = std::nullopt,
        std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::minimal_matmul

namespace ttnn::experimental {
constexpr auto minimal_matmul = ttnn::register_operation<
    "ttnn::experimental::minimal_matmul",
    ttnn::operations::experimental::minimal_matmul::ExecuteMinimalMatmul>();
}
