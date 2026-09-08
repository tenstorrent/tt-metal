// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// One decode step of the gated delta rule (B = 1), one core per value head. `state` is updated in place.
// With conv_states/conv_taps given, `qkv` is the full projection row [q|k|v|z|a|b], `beta` is dt_bias and `g` is
// -exp(A_log): the 4-tap causal conv + SiLU, beta/decay gates and the silu(z) output gate are computed in-kernel and
// the conv states are shifted in place.
ttnn::Tensor gdn_decode_step(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    const ttnn::Tensor& state,
    const ttnn::Tensor& weight,
    uint32_t num_value_heads,
    uint32_t num_key_heads,
    uint32_t key_dim,
    uint32_t value_dim,
    std::optional<float> scale = std::nullopt,
    float l2_epsilon = 1e-6f,
    float norm_epsilon = 1e-6f,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    ttnn::DataType output_dtype = ttnn::DataType::BFLOAT16,
    const std::optional<std::vector<ttnn::Tensor>>& conv_states = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& conv_taps = std::nullopt,
    uint32_t qkvz_dim = 0);

}  // namespace ttnn::experimental::kda
