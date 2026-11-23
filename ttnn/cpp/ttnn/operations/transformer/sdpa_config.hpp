// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tt-metalium/workload/specification/compute_modes.hpp>
#include <tt-metalium/vocabulary/core_coord.hpp>

namespace ttnn::operations::transformer {

struct SDPAProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;
    // Ring joint chunked prefill only (ignored where Q is as long as the local K/V): bands of cores that split each
    // (head, Q chunk) unit along K; 1 disables. The op refuses a count it cannot apply and names the most it can.
    uint32_t k_splits = 1;
    // Ring joint streaming compute only: fidelity of the QK^T and softmax @ V matmuls; the rest of the kernel keeps
    // the compute kernel config's.
    std::optional<tt::tt_metal::MathFidelity> matmul_math_fidelity;
    // Ring joint chunked prefill only (ignored where Q is as long as the local K/V): per-ring-iteration accumulation,
    // so no bf16 running sum spans the whole prefix. Refused with a K split, a sliding window, or several Q chunks
    // per core (raise q_chunk_size).
    bool segmented_accumulation = false;
    // sdpa, ring joint and exp ring joint SDPA, streaming compute only (fp32_dest_acc_en off; ignored otherwise): the
    // narrowest pack, in tiles, that goes out as one block rather than tile by tile. Unset keeps the architecture
    // default, 8 on Wormhole and 2 on Blackhole.
    std::optional<uint32_t> min_blocked_pack_tiles;
    // Decode only: Q is zero on its first qk_skip_head_dim columns (a multiple of 32, below the K head dim), so the
    // QK^T matmul skips those columns; the result is the same as with 0, which multiplies every column.
    uint32_t qk_skip_head_dim = 0;
    // Decode with non-paged K, no K multicast, V read from K (flash MLA): this K tile column (32 columns) holds the
    // same bytes for every key, so the reader loads it from DRAM only while each K buffer slot is first filled.
    std::optional<uint32_t> const_k_tile_col;
    // Decode only: with half tiles (16-row Q, <= 16 heads) the softmax exp is the accurate one, as with full tiles,
    // instead of the approximate one; the rows are then bit-equal to the full-tile path.
    bool exact_exp_half_tile = false;
    // Decode, causal, speculative verify: Q rows come in groups of rows_per_token, one group per token at consecutive
    // positions (cur_pos = the last token's). Row r's causal bound is cur_pos - (tokens - 1 - r / rows_per_token).
    // Exact when every token's bound lies in the same K chunk as cur_pos (the caller checks); 0 = off.
    uint32_t rows_per_token = 0;
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
