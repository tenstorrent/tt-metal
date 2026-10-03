// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::experimental {

// Fused RMSNorm + unary activation (e.g. SiLU, GELU). Equivalent to:
//   tensor = ttnn.rms_norm(input, ...)
//   ttnn.<activation>(tensor)
// but computed in a single kernel pass.
ttnn::Tensor dit_rms_norm_unary_fused(
    const ttnn::Tensor& input_tensor,
    float epsilon = 1e-5f,
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt,
    const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation = std::nullopt);

// Same as dit_rms_norm_unary_fused but also returns the pre-add sum (input + residual). Lets a resnet
// block fuse its terminal add into the next block's norm: the next norm consumes (h, residual) and
// emits both the normed result and the materialized sum the skip connection needs. residual_input_tensor
// is required. Returns {normed, sum}.
std::tuple<ttnn::Tensor, ttnn::Tensor> dit_rms_norm_unary_fused_residual_sum(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& residual_input_tensor,
    float epsilon = 1e-5f,
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const ttnn::prim::LayerNormProgramConfig>& program_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation = std::nullopt);

}  // namespace ttnn::experimental
