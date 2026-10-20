// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Page-table format of a paged KV cache, shared by the ops that write it (update_padded_kv_cache) and read it
// (high_bw_all_gather), host and device side.
//
// The pool holds page-sized cache entries [num_pages * num_layers, heads, page, head_dim]; physical page p of layer l
// is entry p * num_layers + l. A request's row of the INT32 table maps every GLOBAL logical page q (sequence positions
// [q * page, (q + 1) * page)) to a physical page in the pool of the rank that holds q, or -1 when unmapped. Positions
// are placed block-cyclically over the sp ranks of the cluster axis in runs of chunk_local, the per-chip input seq of a
// write, so q lives on rank (q * page / chunk_local) % sp and every write and read of a request must use the same
// chunk_local. One replicated row serves the whole mesh; each chip reads only its own entries.
//
// This is not the page-table format of paged_update_cache / SDPA, whose INT32 entries are device-local blocks of a
// per-device, per-layer cache.
namespace ttnn::operations::experimental::paged_kv {

// The unmapped entry (-1) as the kernels read it.
inline constexpr uint32_t kUnmappedPage = 0xFFFFFFFFu;

// The global row (token, or tile-row when tiled) a chip-local cache row holds under block-cyclic placement.
constexpr uint32_t global_row_of_local(uint32_t local_row, uint32_t rank, uint32_t ranks, uint32_t chunk_rows) {
    return ((local_row / chunk_rows) * ranks + rank) * chunk_rows + local_row % chunk_rows;
}

// Table entries one writer window holds: one slab's run of at most chunk_rows / page_rows pages (a page never straddles
// a slab), read in whole groups of `group` DRAM-aligned entries, plus one group of slack for the aligned start.
constexpr uint32_t window_capacity(uint32_t chunk_rows, uint32_t page_rows, uint32_t group) {
    return ((chunk_rows / page_rows + group - 1) / group + 1) * group;
}

}  // namespace ttnn::operations::experimental::paged_kv
