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
