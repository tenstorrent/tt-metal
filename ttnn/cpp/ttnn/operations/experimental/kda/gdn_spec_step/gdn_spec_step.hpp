// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

// Fused GDN speculative-verify step: conv window rebuild + depthwise causal conv + SiLU + l2norms + gates + T-step
// gated delta rule with the state after every token written out + gated RMSNorm + silu(z) gate, one core per (user,
// value head). ring (or the pages ring_page_idx names) and win_a / win_b are updated in place; the ctrl page carries
// the window parity, mi and the start state per (u,h) (HOLD sentinel 0xFFFFFFFF: no state writes for that head, window
// rows copied through). fast_mode / fp32_state select the per-token arithmetic of gdn_decode_step(fast_mode,
// fp32_state), so a T-token call equals T chained gdn_decode_step calls at B = 1.
// value_blocks and kernel_opts select bit-identical faster programs (see GdnSpecStepParams). hnew_depth and max_users
// accept only hnew_depth 2 and max_users 0 or B.
ttnn::Tensor gdn_spec_step(
    const ttnn::Tensor& qkvzab,
    const std::optional<ttnn::Tensor>& win_a,
    const std::optional<ttnn::Tensor>& win_b,
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
    ttnn::DataType output_dtype = ttnn::DataType::BFLOAT16,
    bool fast_mode = false,
    bool fp32_state = false,
    uint32_t max_users = 0,
    uint32_t value_blocks = 1,  // cores per (u, h) state (column split, bit-identical); clamped to the grid
    uint32_t kernel_opts = 0,   // GdnSpecStepKernelOpt bits (fast_mode + fp32_state only)
    const std::optional<ttnn::Tensor>& ring_page_idx = std::nullopt,  // [>= B, T*Nv]: per-token, per-head ring pages
    const std::optional<ttnn::Tensor>& ring_pool2 = std::nullopt,     // second fp32 pool past ring's pages
    const std::optional<ttnn::Tensor>& conv_hist = std::nullopt);     // packed history pages (no window pair)

}  // namespace ttnn::experimental::kda
