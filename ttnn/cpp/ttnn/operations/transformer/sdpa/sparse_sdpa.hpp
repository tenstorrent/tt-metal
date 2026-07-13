// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::transformer {

// Sparse MLA prefill (DeepSeek DSA), Blackhole single-chip.
//   q       [1, H, S, K_DIM] bf16 or fp8_e4m3 ROW_MAJOR  (H = head count, any multiple of 32; K_DIM = head
//                                                          dim, e.g. 576)
//   kv      [1, 1, T, K_DIM] bf16 or fp8_e4m3 ROW_MAJOR  (K = full K_DIM, V = kv[..., :v_dim]; fp8 halves
//                                                          the K-gather bytes, tilized in-op to bfp8_b)
//   indices [1, 1, S, TOPK] uint32 ROW_MAJOR (0xFFFFFFFF = masked; sentinels are a contiguous tail)
//   v_dim   width of V (leading v_dim cols of the K_DIM-wide cache); the output width.
// Returns out [1, H, S, v_dim] ROW_MAJOR; the output dtype MATCHES q (bf16 q -> bf16 out, fp8 q -> fp8 out).
// (K_DIM is taken from q/kv; scale defaults to K_DIM**-0.5.)
//
// cache_batch_idx: when set, kv is a shared [B, 1, T, K_DIM] cache and this selects the batch slot to attend
// to (indices are page ids within that slot). kv may then also be ND-sharded across DRAM banks. The value is
// a dynamic runtime arg, so changing the slot (or the cache length T) does not recompile the kernels.
//
// block_cyclic_sp_axis / block_cyclic_chunk_local: when both set, `indices` are NATURAL token positions but kv
// is stored block-cyclic across an SP-sharded cache (the DeepSeek chunked-prefill KVPE cache). The gather
// kernels remap each index natural -> physical page on the fly, so the caller does NOT need to reorder kv back
// to natural order. Both must be set together.
//   block_cyclic_sp_axis    : the MESH axis the cache was striped over. `sp` is read from mesh shape on that
//                             axis (the op derives it, so a caller cannot pass an sp inconsistent with the
//                             device); T % sp == 0 required.
//   block_cyclic_chunk_local: the per-shard chunk length (chunk_size_global / sp). Cross-checked at the entry
//                             against q's per-chip seq length: must equal q_isl or tp*q_isl (tp = mesh/sp),
//                             the only two values it can legally take (post-reshard q is sliced by tp).
//
// Producer preconditions (NOT validated per-element): sentinels are a contiguous tail, every row has >= 1
// valid key, and all non-sentinel indices are < T.
ttnn::Tensor sparse_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    uint32_t v_dim,
    std::optional<float> scale = std::nullopt,
    uint32_t k_chunk_size = 128,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> block_cyclic_sp_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_chunk_local = std::nullopt);

// qr-ring Q-gather variant: same op, but ALSO returns the per-(head,query) softmax stats so per-SP-shard
// outputs can be flash-merged (online softmax) across a stationary-KV ring. Returns {O, m, l}:
//   O [1, H, S, v_dim]  — the normalized per-shard output (identical to sparse_sdpa's output)
//   m [1, H, S, 32]     — raw row-max (col 0; UNSCALED max of q·k over the shard's selected keys)
//   l [1, H, S, 32]     — softmax denominator (col 0; sum exp(scale*(q·k - m)))
// Merge across shards i: M=max_i(scale*m_i); w_i=exp(scale*m_i - M)*l_i; out = sum_i w_i O_i / sum_i w_i.
std::vector<ttnn::Tensor> sparse_sdpa_stats(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    uint32_t v_dim,
    std::optional<float> scale = std::nullopt,
    uint32_t k_chunk_size = 128,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> block_cyclic_sp_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_chunk_local = std::nullopt);

// qr-ring SHARD-LOCAL building block. kv is ONLY this rank's stationary block-cyclic stripe [1,1,T/sp,K_DIM];
// `shard` (0..sp-1) says which. Keeps the natural indices that land in this stripe ((n/chunk_local)%sp==shard),
// remaps them to LOCAL pages, and computes the per-shard sparse partial + stats {O, m, l}. A query that
// selected no keys from this stripe yields the identity partial (O=0, m=-BIG, l=0). sp/chunk_local/shard are
// EXPLICIT (not mesh-derived), so it runs single-device for validating the per-shard + flash-merge decomposition.
// Merge across shards i: M=max_i(scale*m_i); w_i=exp(scale*m_i - M)*l_i; out = sum_i w_i O_i / sum_i w_i.
std::vector<ttnn::Tensor> sparse_sdpa_stats_shard_local(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& shard_id,  // [1,1,1,1] uint32, SP-sharded so device s holds its stripe id s
    uint32_t v_dim,
    uint32_t sp,
    uint32_t chunk_local,
    std::optional<float> scale = std::nullopt,
    uint32_t k_chunk_size = 128,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
