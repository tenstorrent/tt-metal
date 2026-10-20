// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Types of the fused DSA (DeepSeek sparse attention) decode state maintenance: from the decode stem matmul
// output, the q_a RMSNorm (multicore) and, per user, the latent append (RMSNorm'd kv_a row into its paged latent
// cache), the indexer ring update (LayerNorm'd k and raw gate rows into the per-user rings) and the pooled key
// of the current pool (softmax-weighted mean over the ring) into the pool cache -- all in place, one program.

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct DsaDecodeStateParams {
    uint32_t B;            // users (rows of the stem)
    uint32_t Lq;           // q_lora_rank (q_a width)
    uint32_t L;            // kv_lora_rank (latent width)
    uint32_t D;            // index head dim (k / gate width)
    uint32_t P;            // pool size (ring slots per pool)
    uint32_t pool_blocks;  // 32-pool blocks per user in the pool cache (identity page table)
    uint32_t qa_off;       // stem column offsets (elements)
    uint32_t kva_off;
    uint32_t k_off;
    uint32_t g_off;
    float eps_rms;  // q_a / kv_a RMSNorm eps
    float eps_ln;   // indexer k LayerNorm eps
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// Shapes (interleaved DRAM/L1):
//   stem         [1, 1, B, Ws]              bf16 TILE    decode stem matmul output; columns [qa | kva | k | g | ...]
//   latent_cache [n_blocks, 1, 32, L]       bf16 TILE    paged latent cache, updated IN PLACE
//   latent_pt    [B, blocks_per_seq]        int32 RM     page table of the latent cache
//   positions    [B]                        int32 RM     position a of every user's token (latent row a, pool a / P,
//                                                        ring slot a % P)
//   ring_k/ring_g[>= B, 1, 32, D]           bf16 TILE    per-user rings (block = user), updated IN PLACE
//   pool_cache   [B * pool_blocks, 1, 32, D] bf16 TILE   pooled keys (identity page table), updated IN PLACE
//   qa_norm_w    [1, 1, 1, Lq] fp32 TILE,  kva_norm_w [1, 1, 1, L],  k_norm_w / k_norm_b [1, 1, 1, D]
//   ape          [1, 1, 32, D]              fp32 TILE    rows < P: the in-pool position embedding, rows >= P: -1e9
//   selectors    [33*32, 32]                bf16 TILE    tile r < 32: 1 at (0, r); tile 32: all ones
// Output: q_resid [1, 1, B, Lq] bf16 TILE; the four caches are returned as-is (aliased).
struct DsaDecodeStateInputs {
    Tensor stem;
    Tensor latent_cache;
    Tensor latent_page_table;
    Tensor positions;
    Tensor ring_k;
    Tensor ring_g;
    Tensor pool_cache;
    Tensor qa_norm_w;
    Tensor kva_norm_w;
    Tensor k_norm_w;
    Tensor k_norm_b;
    Tensor ape;
    Tensor selectors;
};

}  // namespace ttnn::prim
