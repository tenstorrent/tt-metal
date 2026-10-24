// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// One decode step of the gated delta rule, one core per value head; `state` ([Bmax, Nv, Dk, Dv]) is updated in place.
// Plain mode: B = 1, `qkv` is [1, 1, 2*Nk*Dk + Nv*Dv] post conv+silu. With conv_hist/conv_taps given (fused-conv mode,
// 1 <= B <= 32): `qkv` is the projection rows [1, B, W] = [q|k|v|z|a|b], `beta` is dt_bias and `g` is -exp(A_log)
// (one row of Nv values each); conv_hist is packed [Bmax, Nv, 4, 32, 32] bf16 (user u's channel chunk c of the head's
// [q|k|v] in tile row 2c + (u & 1), slot 3 newest) and conv_taps [Nv, 4, 32, 32] bf16 (chunk c in rows 2c and 2c + 1,
// tap 0 oldest). The 4-tap causal conv + SiLU, beta/decay gates and the silu(z) output gate are computed in-kernel and
// the packed history is shifted in place. Output [1, B, Nv*Dv]; only rows 0..B-1 are valid. fast_mode (fused-conv
// mode only) selects the faster *_conv_fast kernels (same o; the in-place state keeps the full fp32 sum, so it
// differs from the default kernels in the last bits).
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
    bool fp32_state = false);

}  // namespace ttnn::experimental::kda
