// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

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
namespace unified_routed_expert_ffn {

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
};

}  // namespace unified_routed_expert_ffn
