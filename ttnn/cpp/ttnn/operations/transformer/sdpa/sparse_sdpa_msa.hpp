// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::transformer {

// MSA block-sparse prefill (MiniMax Sparse Attention), Blackhole single-chip. Attends the block_size-token KV
// blocks named in `indices`; -1 sentinels mask the tail. K and V are separate tensors, and RoPE/QK-norm must be
// applied upstream.
//   q       [1, H, S, d]         bf16 | fp8_e4m3   ROW_MAJOR  (d = head dim, e.g. 128)
//   k       [B, n_kv, T, d]      bf16 | bfp8_b     TILE  (pre-tiled, block-aligned; may be ND-sharded)
//   v       [B, n_kv, T, v_dim]  bf16 | bfp8_b     TILE  (separate tensor; v_dim = the output width, taken from v)
//   indices [1, n_kv, S, TOPK]   uint32 BLOCK-ids  ROW_MAJOR  (-1 = 0xFFFFFFFF = sentinel; a contiguous tail)
// Returns [1, H, S, v_dim] ROW_MAJOR with dtype matching q. `scale` defaults to d**-0.5; `block_size`
// defaults to 128. H must be divisible by n_kv; H/n_kv may be 16 (internally padded to one head tile) or a
// multiple of 32.
//
// Preconditions: d, v_dim, and block_size are multiples of 32; block_size divides T; TOPK*4 and output row
// bytes meet DRAM alignment; each index row has at least one valid block; all valid block ids are < T/block_size.
//
// `chunk_start_idx` needs to be passed (the global position of query row 0) to enforce a token-level causal
// mask on the diagonal block — required for correct causal prefill, where a query's own block holds future
// tokens that must not be attended. Unset (default) can be used for full attention.
// `cluster_axis` derives the per-device start under sequence parallelism (chunk_start = chunk_start_idx + rank*S).
// Causal masking requires bf16 q: with fp8 q the mask is numerically wrong (fp8-specific;
// root cause not identified), so fp8 q with `chunk_start_idx` asserts.
ttnn::Tensor sparse_sdpa_msa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& indices,
    std::optional<float> scale = std::nullopt,
    uint32_t block_size = 128,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttnn::transformer
