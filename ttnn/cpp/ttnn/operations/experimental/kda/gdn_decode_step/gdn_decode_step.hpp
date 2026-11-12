// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// One gated delta rule decode step for B <= 32 batch rows, one core per (value head, group of rows). `qkv` holds the
// projection rows [q|k|v|z|a|b], `beta` is dt_bias and `g` is -exp(A_log). The op computes the 4-tap causal conv + SiLU
// from the packed history conv_hist and the packed taps conv_taps (both required), the beta/decay gates, the recurrence
// and the silu(z)-gated RMSNorm output, and updates `state` and conv_hist in place. A packed [32, 32] bf16 tile holds
// channel chunk c of the head's [q|k|v] in row 2c + (b & 1) for batch row b (conv_taps: in rows 2c and 2c + 1).
// fast_mode selects the faster *_conv_fast kernels (same o; the in-place state keeps the full fp32 sum, so it differs
// from the default kernels in the last bits). state_page_idx makes `state` and `conv_hist` page pools: column b of the
// [2*Nv, >= B] index names, per value head, the first state page and the first conv_hist page of batch row b. The
// optional second pools extend each page space: index pages past the first pool's page count address state_pool2 /
// conv_hist_pool2.
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
    const std::optional<ttnn::Tensor>& conv_hist = std::nullopt,
    const std::optional<ttnn::Tensor>& conv_taps = std::nullopt,
    uint32_t qkvz_dim = 0,
    bool fast_mode = false,
    bool fp32_state = false,
    const std::optional<ttnn::Tensor>& state_page_idx = std::nullopt,
    const std::optional<ttnn::Tensor>& state_pool2 = std::nullopt,
    const std::optional<ttnn::Tensor>& conv_hist_pool2 = std::nullopt);

}  // namespace ttnn::experimental::kda
