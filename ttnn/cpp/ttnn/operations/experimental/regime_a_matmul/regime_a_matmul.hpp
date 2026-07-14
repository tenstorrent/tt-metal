// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_device_operation_types.hpp"

namespace ttnn::operations::experimental::regime_a_matmul {

// Re-export the config type.
using RegimeAMatmulConfig = ttnn::experimental::prim::RegimeAMatmulConfig;

}  // namespace ttnn::operations::experimental::regime_a_matmul

namespace ttnn::experimental {

ttnn::Tensor regime_a_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
