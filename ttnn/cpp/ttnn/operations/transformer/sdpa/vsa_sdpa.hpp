// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <optional>

namespace ttnn::transformer {

// VSA (video sparse attention) fine-stage block-sparse attention, Blackhole single-chip, non-causal.
// Forked from sparse_sdpa_msa. Per (head, 64-token query tile), attends exactly the KV blocks named in that
// row of `indices`, with online softmax numerically equivalent to dense SDPA under the same block mask.
// RoPE/QK-norm must be applied upstream. The kernel has no SP/mesh logic: the index list is the whole contract.
//   q            [1, H, S, d]      bf16          TILE (head-major; S a multiple of 64)
//   k            [1, H, T, d]      bf16 | bfp8_b TILE (T a multiple of block_size)
//   v            [1, H, T, d]      bf16 | bfp8_b TILE
//   indices      [1, H, S/64, W]   uint32        ROW_MAJOR global block ids; 0xFFFFFFFF sentinel tail;
//                                                W >= T/block_size (pad W with sentinels for DRAM alignment)
//   block_counts [1, 1, 1, W]      uint32        ROW_MAJOR valid tokens per block, in (0, block_size];
//                                                key columns >= count are masked to -inf (zero-padded tiles)
// Returns [1, H, S, d] TILE bf16.
//
// `scale` defaults to d**-0.5. `block_size` is the VSA tile size (default 64, the (4,4,4) cube).
// `k_chunk_blocks` (m >= 1) is the k-chunk multiplier: the reader gathers a row's next m listed blocks into
// one contiguous L1 chunk and compute does one QK matmul and one softmax-rescale per chunk; a row whose valid
// block count is not a multiple of m ends with a partial chunk. Results are identical for every m.
//
// Preconditions: each index row has at least one valid block; all valid ids < T/block_size; W*4 bytes meets
// DRAM row alignment.
ttnn::Tensor vsa_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& block_counts,
    std::optional<float> scale = std::nullopt,
    uint32_t block_size = 64,
    uint32_t k_chunk_blocks = 1,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace ttnn::transformer
