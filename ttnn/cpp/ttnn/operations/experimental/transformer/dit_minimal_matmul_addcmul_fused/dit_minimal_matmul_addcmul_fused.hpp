// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::experimental {

// Fused minimal_matmul + addcmul. Calls minimal_matmul with fused ternary (addcmul) parameters.
ttnn::Tensor dit_minimal_matmul_addcmul_fused(
    const ttnn::Tensor& matmul_input_tensor,
    const ttnn::Tensor& matmul_weight_tensor,
    float scalar,
    const ttnn::Tensor& addcmul_input_tensor1,
    const ttnn::Tensor& addcmul_input_tensor2,
    const std::optional<ttnn::Tensor>& bias_tensor = std::nullopt,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
