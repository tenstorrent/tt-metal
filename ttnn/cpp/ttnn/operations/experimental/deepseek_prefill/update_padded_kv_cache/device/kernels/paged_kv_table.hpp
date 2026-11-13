// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Page-table format of a paged KV cache, host and device side.
//
// Every chip holds its own pool of single-head pages [num_pages * num_layers, 1, page, head_dim]; physical page p of
// layer l is entry pool_entry(p, l, num_layers). A request's row of the INT32 table maps every GLOBAL logical page q
// (sequence positions [q * page, (q + 1) * page)) to a physical page in the pool of the chip at SP coordinate
// (q * page / chunk_local) % sp along the cluster axis, or -1 when unmapped: positions are placed block-cyclically over
// the sp coordinates in slabs of chunk_local (the rows a write puts on each chip: its input row window when it has one,
// else the per-chip input seq), so every write and read of a request must use the same chunk_local.
// One replicated row serves the whole mesh; each chip reads only its own entries.
//
// This is not the page-table format of paged_update_cache / SDPA, whose INT32 entries are device-local blocks of a
// per-device, per-layer cache.
namespace ttnn::operations::experimental::paged_kv {

// The unmapped entry (-1) as the kernels read it.
inline constexpr uint32_t kUnmappedPage = 0xFFFFFFFFu;

// The pool entry holding physical page `page` of `layer`.
constexpr uint32_t pool_entry(uint32_t page, uint32_t layer, uint32_t num_layers) { return page * num_layers + layer; }

// The global row (token, or tile-row when tiled) a chip-local cache row holds under block-cyclic placement.
constexpr uint32_t global_row_of_local(uint32_t local_row, uint32_t rank, uint32_t ranks, uint32_t chunk_rows) {
    return ((local_row / chunk_rows) * ranks + rank) * chunk_rows + local_row % chunk_rows;
}

// Table entries one writer table window holds: one slab's run of at most chunk_rows / page_rows pages (a page never
// straddles a slab), read in whole groups of `group` DRAM-aligned entries, plus one group of slack for the aligned
// start.
constexpr uint32_t table_window_capacity(uint32_t chunk_rows, uint32_t page_rows, uint32_t group) {
    return ((chunk_rows / page_rows + group - 1) / group + 1) * group;
}

}  // namespace ttnn::operations::experimental::paged_kv
