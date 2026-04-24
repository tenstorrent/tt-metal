// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail {

// Largest divisor of n that is <= max_val
constexpr uint32_t largest_divisor(uint32_t n, uint32_t max_val) {
    for (uint32_t d = max_val; d >= 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

// Find best in0_block_w: largest divisor of K_tiles that keeps estimated CB usage within
// the device's actual L1 budget, queried via matmul utilities (same approach as matmul op).
uint32_t best_in0_block_w(
    uint32_t K_tiles,
    uint32_t per_core_M,
    uint32_t per_core_N,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    tt::tt_metal::DataType output_dtype,
    float l1_safety_margin = 0.9f);

ttnn::Tensor routed_expert_ffn_default(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output);

// Wormhole-optimized path (8x8 grid)
ttnn::Tensor routed_expert_ffn_wh(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output);

// Blackhole-optimized path (14x10 grid). When both global_expert_idx_table and
// expert_token_counts are provided, each matmul dispatches via the forked routed
// device op and evaluates the per-chunk guard on-device:
//   skip iff expert_token_counts[global_expert_idx_table[local_expert_idx]]
//         <= curr_expert_iter * expert_iter_length.
// When either tensor is nullopt, falls back to plain ttnn::matmul (the scalars
// are ignored in that case).
ttnn::Tensor routed_expert_ffn_bh(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> output,
    const std::optional<ttnn::Tensor>& global_expert_idx_table,
    const std::optional<ttnn::Tensor>& expert_token_counts,
    uint32_t local_expert_idx,
    uint32_t curr_expert_iter,
    uint32_t expert_iter_length);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::detail
