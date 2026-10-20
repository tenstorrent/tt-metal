// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Page-table format of a paged KV cache, shared by the host op and the device kernels.
//
// A request's row maps every GLOBAL logical page q (sequence positions [q * page, (q + 1) * page)) to a physical page
// in the pool of the chip that holds q. Block-cyclic SP placement puts q on rank (q * page / chunk_local) % sp, so one
// replicated row serves the whole mesh and each chip reads only its own entries. The pool holds page-sized cache
// entries, [num_pages * num_layers, heads, page, head_dim], with physical page p of layer l at entry p * num_layers +
// l.
namespace ttnn::operations::experimental::deepseek_prefill::paged_kv_table {

inline constexpr uint32_t kUnmappedPage = 0xFFFFFFFFu;

// Entries per DRAM-aligned table read (Blackhole DRAM reads are 64-byte aligned); rows are a whole number of these.
inline constexpr uint32_t kTableEntriesPerRead = 16;

// L1 entries for one window of a row: one write covers at most chunk_rows / page_rows pages of a slab, one more when it
// starts mid-page, plus the alignment slack around the read.
constexpr uint32_t window_capacity(uint32_t chunk_rows, uint32_t page_rows) {
    return ((chunk_rows / page_rows + 1 + kTableEntriesPerRead - 1) / kTableEntriesPerRead + 1) * kTableEntriesPerRead;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::paged_kv_table
