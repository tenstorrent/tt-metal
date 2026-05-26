// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

// Single-op fused routed-expert FFN for DeepSeek V3 prefill on Blackhole.
//
// Performs, all inside ONE device program (one row in tt-perf-report):
//   gate = matmul(x, gate_proj)
//   up   = matmul(x, up_proj)
//   y    = matmul(silu(gate) * up, down_proj)
//
// The kernel chunks the M axis internally and reads the device-resident
// `counts[global_expert_idx_table[local_expert_id]]` value at runtime to skip
// chunks beyond the actual token count for this expert.
//
// Args:
//   x: (M_max, K=emb), TILE, DRAM interleaved, BFLOAT8_B/BFLOAT16. Only the
//      first ceil_tile(count) rows are valid (the rest is dispatch padding).
//   gate_proj: (K=emb, N=hidden), TILE, DRAM interleaved (any weights dtype).
//   up_proj:   (K=emb, N=hidden), TILE, DRAM interleaved (any weights dtype).
//   down_proj: (K=hidden, N=emb), TILE, DRAM interleaved (any weights dtype).
//   counts: device-resident UINT32 vector, one entry per global expert id.
//   global_expert_idx_table: device-resident UINT32 vector,
//      counts[global_expert_idx_table[local_expert_id]] == this expert's
//      actual token count.
//   local_expert_id: index into global_expert_idx_table.
//   compute_kernel_config: optional matmul math fidelity / accumulator config.
//   output: optional pre-allocated (M_max, K=emb) DRAM-interleaved output
//      tensor to write into. Must match x.dtype() and shape.
ttnn::Tensor unified_routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    const ttnn::Tensor& expert_region_offsets,
    uint32_t local_expert_id,
    bool use_region_offsets = true,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& output = std::nullopt);

// MoE-level composite op: takes the dispatched buffer + ALL experts' weights
// and loops experts_per_chip times INSIDE the op, calling
//   extract -> unified_routed_expert_ffn -> insert
// per expert. Python passes everything; no host-side counts/idx read, no
// per-expert Python loop. The unified FFN reads counts on-device so each
// expert's work scales to its actual count.
ttnn::Tensor unified_routed_expert_moe(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn

namespace ttnn {
using operations::experimental::deepseek_prefill::unified_routed_expert_ffn::unified_routed_expert_ffn;
using operations::experimental::deepseek_prefill::unified_routed_expert_ffn::unified_routed_expert_moe;
}  // namespace ttnn
