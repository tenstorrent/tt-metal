// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "minimal_matmul.hpp"
#include "device/minimal_matmul_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::minimal_matmul {

ttnn::Tensor ExecuteMinimalMatmul::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<unary::UnaryWithParam> fused_activation,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::minimal_matmul(
        input_tensor,
        weight_tensor,
        bias_tensor,
        std::move(fused_activation),
        config,
        memory_config,
        dtype,
        compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::minimal_matmul
