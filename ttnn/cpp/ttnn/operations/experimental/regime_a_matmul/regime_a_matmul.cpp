// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "regime_a_matmul.hpp"

#include "device/regime_a_matmul_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental {

ttnn::Tensor regime_a_matmul(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<float> fused_ternary_scalar,
    const std::optional<ttnn::Tensor>& fused_ternary_input_a,
    const std::optional<ttnn::Tensor>& fused_ternary_input_b,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    auto outs = ttnn::prim::regime_a_matmul(
        input_tensor,
        weight_tensor,
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        bias_tensor,
        std::move(fused_activation),
        fused_ternary_scalar,
        fused_ternary_input_a,
        fused_ternary_input_b,
        1,    // chunks
        -1);  // dim
    TT_FATAL(outs.size() == 1, "regime_a_matmul expected a single output, got {}", outs.size());
    return outs[0];
}

std::vector<ttnn::Tensor> regime_a_matmul_split(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    int32_t chunks,
    int32_t dim,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<ttnn::Tensor>& bias_tensor,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    std::optional<float> fused_ternary_scalar,
    const std::optional<ttnn::Tensor>& fused_ternary_input_a,
    const std::optional<ttnn::Tensor>& fused_ternary_input_b,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(chunks >= 1, "regime_a_matmul_split requires chunks >= 1, got {}", chunks);
    TT_FATAL(dim == -1, "regime_a_matmul_split only supports dim=-1, got {}", dim);
    return ttnn::prim::regime_a_matmul(
        input_tensor,
        weight_tensor,
        config,
        memory_config,
        dtype,
        compute_kernel_config,
        bias_tensor,
        std::move(fused_activation),
        fused_ternary_scalar,
        fused_ternary_input_a,
        fused_ternary_input_b,
        chunks,
        dim);
}

}  // namespace ttnn::experimental
