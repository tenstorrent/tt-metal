// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

// Activation/glu mode. SILU is the original DeepSeek path (gate matmul fuses
// silu, multiply with up, then down). GPT_OSS_SWIGLU runs gate/up matmuls with
// no fused activation and applies the GPT-OSS clamped SwiGLU SFPU between them.
enum class RoutedExpertMode : uint32_t {
    SILU = 0,
    GPT_OSS_SWIGLU = 1,
};

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    RoutedExpertMode mode = RoutedExpertMode::SILU,
    const std::optional<const ttnn::Tensor>& gate_bias = std::nullopt,
    const std::optional<const ttnn::Tensor>& up_bias = std::nullopt,
    const std::optional<const ttnn::Tensor>& down_bias = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<ttnn::Tensor> output = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::routed_expert_ffn::routed_expert_ffn;
using operations::experimental::deepseek_prefill::routed_expert_ffn::RoutedExpertMode;
}  // namespace ttnn
