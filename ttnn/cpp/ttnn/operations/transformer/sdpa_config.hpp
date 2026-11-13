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
    // Ring joint only: with fp32_dest_acc_en, run the streaming compute path (fp32 dest accumulation and fp32 running
    // sums, bf16 QK^T score block) instead of the compute_common.hpp path. Not bit-identical to that path (the scores
    // round to bf16 before max / exp); faster, and closer to an fp64 reference. No effect without fp32_dest_acc_en
    // (that already streams); ignored by the other SDPA ops.
    bool streaming_fp32_dest = false;
    // Ring joint only: also run SDPA on the cores (x .. grid.x - 1, grid.y) of the CCL row below the SDPA grid that the
    // fused all-gather does not use (it takes num_links * 2 cores from ccl_core_grid_offset); e.g. 12 x 9 + 8 = 116
    // cores. Only the core a (head, Q chunk) unit runs on changes: outputs are byte-identical. Unset: unchanged.
    std::optional<uint32_t> ccl_row_sdpa_start_x = std::nullopt;
    // Ring joint only: on a program-cache hit, rewrite only the runtime args of buffers whose address changed since
    // the last hit of the same program, instead of every binding. Same argument values. Off: the full rewrite.
    bool changed_bindings_only = false;
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
