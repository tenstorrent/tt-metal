// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Coalesced weight-tile DRAM reads, shared by the reader and the writer so the five weight read
// sites (reader gate/up/down, writer up/down) have ONE definition of the coalescing.
//
// SHARD_W > 0 — the weight tensor is DRAM ND-sharded, SHARD_W tile columns per shard. TensorAccessor
// places page (k, n) inside its shard at `(k % shard_k_rows) * SHARD_W + (n % SHARD_W)`
// (tensor_accessor.h get_bank_and_offset), so for a fixed k the pages of one shard row are
// physically contiguous in ONE bank, strided by the accessor's own page size (tensor_accessor.h
// get_noc_addr) -- which is `page_bytes` here, so `len * page_bytes` spans exactly `len` pages. A
// run therefore extends to the next SHARD_W boundary and leaves as a single NoC transaction. Shard
// HEIGHT never enters: the n-stride inside a shard is SHARD_W regardless of how many k-rows it
// holds.
//
// SHARD_W == 0 — DRAM-interleaved. Consecutive n land in different banks by construction, so no
// contiguous run exists: run() collapses to 1 and this degenerates to the per-tile read.
//
// WeightTable — table-addressed weights (WEIGHT_PAGES). Each tile column of one expert has a table entry
// (dram_bank_id, byte_offset) of its tile k = 0; tile k sits k_stride bytes further in the same bank. byte_offset
// is relative to the source tensor's bank base address (BankView). Columns are independent placements, so there
// is no run across n: one read per tile. The same read() call sites take either source.
namespace unified_routed_expert_ffn {

template <class Acc>
struct WeightTable {
    BankView<Acc> src;    // the source tensor's banks
    uint32_t entries_l1;  // L1 address of entry 0 of this role (column j's entry: 2 words at entries_l1 + 8 j)
    uint32_t k_stride;    // bytes from tile k to tile k + 1 of one column
};

// entries_l1 points at this role's first entry in the expert's weight_pages row (already in L1).
template <class Acc>
FORCE_INLINE WeightTable<Acc> make_weight_table(const Acc& src, uint32_t entries_l1, uint32_t k_stride) {
    return WeightTable<Acc>{BankView<Acc>(src), entries_l1, k_stride};
}

template <uint32_t SHARD_W = 0>
struct WeightRuns {
    // Length of the maximal contiguous run starting at tile column j, clipped to end.
    static FORCE_INLINE uint32_t run(uint32_t j, uint32_t end) {
        if constexpr (SHARD_W > 0) {
            const uint32_t to_shard_edge = SHARD_W - (j % SHARD_W);
            const uint32_t remaining = end - j;
            return (to_shard_edge < remaining) ? to_shard_edge : remaining;
        } else {
            return 1;
        }
    }

    // Read tile columns [j0, jend) of tensor tile-row `row`, into L1 at
    // `l1_base + (j - j0) * page_bytes`. Columns are the free dim, so callers pass jend clipped to
    // the tensor's real N and the phantom columns past it stay UNWRITTEN — the guard the per-tile
    // loops used to spell out per column. Reads are ISSUED only; the caller owns the barrier.
    template <class Acc>
    static FORCE_INLINE void read(
        const Noc& noc,
        const Acc& acc,
        uint32_t row,
        uint32_t n_tiles_full,
        uint32_t j0,
        uint32_t jend,
        uint32_t l1_base,
        uint32_t page_bytes) {
        const uint32_t page_row_base = row * n_tiles_full;
        uint32_t j = j0;
        uint32_t off = 0;
        while (j < jend) {
            const uint32_t len = run(j, jend);
            // The accessor supplies the run's START page and the size covers the whole run, so
            // `len` pages leave as one transaction rather than one per page.
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(l1_base + off * page_bytes),
                len * page_bytes,
                {.page_id = page_row_base + j},
                {});
            j += len;
            off += len;
        }
    }

    // Table-addressed form of read(): tile column j of K-row `row` comes from its own table entry. Same L1 layout
    // and the same phantom-column rule as above (columns past jend stay unwritten). Reads are ISSUED only.
    template <class Acc>
    static FORCE_INLINE void read(
        const Noc& noc,
        const WeightTable<Acc>& table,
        uint32_t row,
        uint32_t /*n_tiles_full*/,
        uint32_t j0,
        uint32_t jend,
        uint32_t l1_base,
        uint32_t page_bytes) {
        const volatile tt_l1_ptr uint32_t* entry =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(table.entries_l1) + 2 * j0;
        const uint32_t row_offset = row * table.k_stride;
        uint32_t l1 = l1_base;
        for (uint32_t j = j0; j < jend; ++j) {
            noc.async_read(
                table.src,
                CoreLocalMem<uint32_t>(l1),
                page_bytes,
                {.bank_id = entry[0], .offset_bytes = entry[1] + row_offset},
                {});
            entry += 2;
            l1 += page_bytes;
        }
    }
};

}  // namespace unified_routed_expert_ffn
