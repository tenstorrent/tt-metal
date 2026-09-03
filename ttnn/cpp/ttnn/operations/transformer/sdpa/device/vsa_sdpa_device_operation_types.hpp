// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

// Primitive for VSA (video sparse attention) block-sparse fine-stage attention, forked from sparse_sdpa_msa.
// Non-causal only. Per (head, 64-token query tile), attends exactly the KV blocks named in that row of
// `indices`; `block_counts` gives each block's valid token count and columns beyond it are masked to -inf
// (VSA tiles are zero-padded to block_size on host, so ragged blocks carry pad columns that must not attend).
struct VsaSdpaParams {
    float scale = 1.0f;        // compile-time; included in the program hash
    uint32_t block_size = 64;  // tokens per KV block (the VSA cube size)
    // Blocks gathered per L1 chunk (the k_chunk multiplier m): one QK matmul + one softmax rescale per chunk.
    // A row whose valid block count is not a multiple of m ends with a partial chunk. v1 path only.
    uint32_t k_chunk_blocks = 1;
    // Streaming (v2) algorithm: rows stay resident in L1 with running softmax state while each
    // core's union of listed KV blocks streams through ONCE in ascending block order -- eliminating
    // the per-row re-gather that makes v1 DRAM-bound. Numerics: identical contract; online softmax
    // is order-independent, so ascending order is as exact as list order (bf16 rounding differs).
    bool streaming = false;
    // Raw-selection mode (streaming only): `indices` rows are the coarse stage's top-k output as-is;
    // the kernel consumes the first `list_len` entries of each row (0 = the whole width), adds the
    // `exempt_ids` blocks to every row, and gives the q-tile rows flagged in the optional
    // `dense_row_mask` input the full block list. Replaces the host-side prefix/sentinel/dense-blend
    // assembly (~10 layout ops per block).
    uint32_t list_len = 0;
    std::vector<uint32_t> exempt_ids;
    // Padded coarse numbering: the coarse stage may number blocks per SP shard in slots of
    // 2^coarse_slots_shift (tile-aligned pooled gathers) while only coarse_real_per_shard of them are
    // real; a listed id b maps to b - (b >> shift) * (2^shift - real). 0 = ids are already real.
    uint32_t coarse_slots_shift = 0;
    uint32_t coarse_real_per_shard = 0;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct VsaSdpaInputs {
    Tensor q;             // [1,H,S,d]   bf16 TILE, head-major (S multiple of 64)
    Tensor k;             // [1,H,T,d]   bf16|bfp8_b TILE (T multiple of block_size)
    Tensor v;             // [1,H,T,d]   bf16|bfp8_b TILE
    Tensor indices;       // [1,H,S/64,W] uint32 block ids, ROW_MAJOR; 0xFFFFFFFF sentinel tail
    Tensor block_counts;  // [1,1,1,Wc]  uint32 valid tokens per block, Wc >= T/block_size
    // raw-selection mode: [1,1,1,words] uint32 ROW_MAJOR, bit q_tile set -> that q-tile row attends every
    // real block (words*4 bytes must be a multiple of 32); per device, so a tensor rather than an attribute
    std::optional<Tensor> dense_row_mask;
};

}  // namespace ttnn::prim
