// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// Fused GDN spec-verify step: conv window rebuild + depthwise causal conv + SiLU + l2norms + gates + T-step gated
// delta rule with per-token fp32 state ring writes + gated RMSNorm + silu(z) gate, one core per (user, value head).
// ring, win_a/win_b are updated in place; the ctrl page carries parity, mi and the initial ring block per (u,h)
// (HOLD sentinel 0xFFFFFFFF: no ring writes for that head, window rows copied through unchanged).
ttnn::Tensor gdn_spec_step(
    const ttnn::Tensor& qkvzab,
    const ttnn::Tensor& win_a,
    const ttnn::Tensor& win_b,
    const ttnn::Tensor& ring,
    const ttnn::Tensor& ctrl,
    const ttnn::Tensor& taps,
    const ttnn::Tensor& dt_bias,
    const ttnn::Tensor& neg_exp_A,
    const ttnn::Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    uint32_t T,
    uint32_t B,
    uint32_t qkvz_dim,
    uint32_t conv_kernel = 4,
    std::optional<float> scale = std::nullopt,
    float l2_epsilon = 1e-6f,
    float norm_epsilon = 1e-6f,
    uint32_t hnew_depth = 2,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    ttnn::DataType output_dtype = ttnn::DataType::BFLOAT16);

}  // namespace ttnn::experimental::kda
