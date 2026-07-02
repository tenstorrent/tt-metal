// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::deepseek_fused_qkv {

// Fused deepseek_v4_flash decode QKV projection (attention.py `_qkv`) as one op.
//
// Returns `{q, kv}`:
//   q  : [1, 1, num_heads, head_dim]  (per-head RMSNormed + partial-RoPE'd query)
//   kv : [1, 1, 1, head_dim]          (RMSNormed + partial-RoPE'd shared K==V)
//
// `wqa`/`wqb`/`wkv` are DRAM WIDTH_SHARDED weights ([K, N], one width shard per
// DRAM bank). `qa_norm_w`/`kv_norm_w` are the RMSNorm gains. `cos`/`sin` are
// DRAM-interleaved [1, 1, 1, rope_dim] tables (single decode position, broadcast
// over heads); `trans_mat` is the single [32, 32] rotate_half tile.
std::vector<ttnn::Tensor> deepseek_fused_qkv(
    const ttnn::Tensor& hidden,
    const ttnn::Tensor& wqa,
    const ttnn::Tensor& wqb,
    const ttnn::Tensor& wkv,
    const ttnn::Tensor& qa_norm_w,
    const ttnn::Tensor& kv_norm_w,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    float eps,
    uint32_t rope_dim,
    uint32_t num_heads,
    const std::optional<tt::tt_metal::MemoryConfig>& q_mem_config = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& kv_mem_config = std::nullopt,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::experimental::transformer::deepseek_fused_qkv

namespace ttnn::experimental {
using operations::experimental::transformer::deepseek_fused_qkv::deepseek_fused_qkv;
}  // namespace ttnn::experimental
