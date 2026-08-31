// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    // A row whose valid block count is not a multiple of m ends with a partial chunk.
    uint32_t k_chunk_blocks = 1;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct VsaSdpaInputs {
    Tensor q;             // [1,H,S,d]   bf16 TILE, head-major (S multiple of 64)
    Tensor k;             // [1,H,T,d]   bf16|bfp8_b TILE (T multiple of block_size)
    Tensor v;             // [1,H,T,d]   bf16|bfp8_b TILE
    Tensor indices;       // [1,H,S/64,W] uint32 block ids, ROW_MAJOR; 0xFFFFFFFF sentinel tail; W >= T/block_size
    Tensor block_counts;  // [1,1,1,W]   uint32 valid tokens per block (entries past T/block_size unused)
};

}  // namespace ttnn::prim
