// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

// Internal (not python-bound) guarded unary used by routed_matmul as the
// post-activation step. Mirrors routed_matmul's guard semantics: each kernel
// reads two DRAM tables at entry and skips iff
//   expert_token_counts[global_expert_idx_table[local_expert_idx]]
//       <= curr_expert_iter * expert_iter_length.
//
// Scope is intentionally narrow (sharded TILE in-place style, single op). See
// routed_unary_program_factory.hpp for the TT_FATAL-enforced preconditions.
ttnn::Tensor routed_unary(
    const ttnn::Tensor& input,
    const ttnn::operations::unary::EltwiseUnaryWithParam& op,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_token_counts,
    uint32_t local_expert_idx,
    uint32_t curr_expert_iter,
    uint32_t expert_iter_length,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
