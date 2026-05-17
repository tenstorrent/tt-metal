// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<ttnn::Tensor> output = std::nullopt,
    // Optional guard tensors. When both are provided (BH only), each of the three
    // matmuls dispatches via the forked routed_matmul device op, whose per-kernel
    // guard reads expert_token_counts[global_expert_idx_table[local_expert_idx]]
    // and skips the chunk iff that value <= curr_expert_iter * expert_iter_length.
    // When either is nullopt, falls back to ttnn::matmul — identical to pre-routed
    // behavior, and the three scalars below are ignored.
    const std::optional<ttnn::Tensor>& global_expert_idx_table = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_token_counts = std::nullopt,
    uint32_t local_expert_idx = 0,
    uint32_t curr_expert_iter = 0,
    uint32_t expert_iter_length = 0);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::routed_expert_ffn::routed_expert_ffn;
}  // namespace ttnn
