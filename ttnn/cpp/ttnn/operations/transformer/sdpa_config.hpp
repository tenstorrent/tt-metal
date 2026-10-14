// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::transformer {

struct SDPAProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;
    // Ring joint chunked prefill only: up to this many cores may share one (head, Q chunk) unit along K when the
    // units leave the grid idle. 1 disables the split.
    uint32_t max_k_splits = 1;
    // Ring joint streaming compute only: fidelity of the QK^T and softmax @ V matmuls; the rest of the kernel keeps
    // the compute kernel config's.
    std::optional<tt::tt_metal::MathFidelity> matmul_math_fidelity;
    // Ring joint chunked prefill only: each core accumulates each ring iteration separately and merges, so no bf16
    // running sum spans the whole prefix. Ignored where K is split; the op refuses it when a core would hold several
    // Q chunks (raise q_chunk_size).
    bool segmented_accumulation = false;
    // SDPA prefill (standard and chunked) and SDPA decode only: keep the online-softmax state (running sum and
    // output accumulator; in decode also the scores and the tree-reduction exchange) in FP32 and update it on the
    // SFPU from lossless (UnpackToDestFp32) reads. The default path rounds that state through BF16 or 19-bit
    // source registers on every K chunk, so its error grows with context length (relative L2 vs FP64 at 512K:
    // 2-7% decode, ~59% prefill; 0.2-0.3% with this option). Decode also uses full tiles and the accurate
    // exponential; on Blackhole, prefill fuses the softmax scale into a base-2 exponential (2.9e-6 relative
    // error). Needs fp32_dest_acc_en; decode supports causal, tiled-Q attention without mask, sink, sliding
    // window, MLA or speculative positions.
    bool fp32_accumulation = false;
};

// Paired geometry for an HMA-shared paged K/V cache (chunked prefill SDPA and
// paged decode SDPA). When the physical buffer was allocated for a different layer's
// (num_kv_heads, block_size, head_dim) view — e.g. vLLM hybrid kv-cache-groups — the
// reader must address it with this call's block_size / num_kv_heads (Q drives head_dim).
// Presence of the override is the outer std::optional on the op API; both fields are
// required plain values whenever the struct is provided. Default {0, 0} means inactive
// after the optional is collapsed into operation attributes.
struct PagedCacheGeometryOverride {
    uint32_t block_size = 0;
    uint32_t num_kv_heads = 0;

    [[nodiscard]] bool active() const { return block_size != 0 || num_kv_heads != 0; }
};

}  // namespace ttnn::operations::transformer
