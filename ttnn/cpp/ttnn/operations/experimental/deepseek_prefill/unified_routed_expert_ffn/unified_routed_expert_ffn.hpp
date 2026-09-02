// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "device/unified_routed_expert_ffn_types.hpp"  // RoutedExpertActivation

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

// Single-op fused routed-expert MoE FFN for DeepSeek V3 prefill on Blackhole.
//
// Takes the whole dispatched buffer + ALL local experts' weights and runs, in
// ONE device program (one row in tt-perf-report), the full SwiGLU FFN for every
// local expert:
//   gate = matmul(x_e, gate_proj_e)
//   up   = matmul(x_e, up_proj_e)
//   y_e  = matmul(silu(gate) * up, down_proj_e)
//
// The device program's reader/compute/writer kernels loop over the
// experts_per_chip local experts. For each expert `e` they resolve its global
// id (global_expert_idx_table[e]), token count (expert_token_counts[global_id])
// and region offset (expert_region_offsets[global_id]) device-side, read that
// expert's slice of the shared dispatched buffer at the region offset and write
// its output straight into the shared output buffer at the same offset.
//
// Args:
//   dispatched_buffer: (max_dispatch, emb). Blackhole fast path: ROW_MAJOR
//     BFLOAT16 (tilized + bf8-packed in-op, fresh TILE bf8 output). A TILE bf8
//     dispatch buffer is instead written in place.
//   expert_region_offsets: UINT32 per-global-expert region start offsets.
//   expert_token_counts: UINT32 per-global-expert token counts.
//   global_expert_idx_table: UINT32 local-slot -> global-expert-id map.
//   gate_projs/up_projs/down_projs: one (emb, hidden)/(emb, hidden)/(hidden,
//     emb) weight tensor per local expert (all identical shape/dtype).
//   max_dispatched_tokens_per_expert: per-expert M the program is sized for.
//
// Keyword args:
//   compute_kernel_config: optional matmul math fidelity / accumulator config.
//   activation: Silu (default, DeepSeek) or SwiGluOai (clamped, MiniMax-M3 /
//     gpt-oss).
//   gate_biases/up_biases/down_biases: optional per-local-expert biases
//     (gpt-oss), all three lists together or none, one entry per local expert.
//
// Returns:
//   ttnn::Tensor: expert outputs, same shape as dispatched_buffer.
ttnn::Tensor unified_routed_expert_moe(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    RoutedExpertActivation activation = RoutedExpertActivation::Silu,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases = std::nullopt,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation;
using operations::experimental::deepseek_prefill::unified_routed_expert_ffn::unified_routed_expert_moe;
}  // namespace ttnn
