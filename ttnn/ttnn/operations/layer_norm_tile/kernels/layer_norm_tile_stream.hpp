// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared tile-stream pieces of the layer_norm reader and writer.
//
// read_block_chunks: a trid ring over one block's chunks. Chunk k is tagged with id (k % depth) + 1;
// once `depth` chunks are outstanding the oldest is retired with a per-id barrier, so the remaining
// chunks stay on the wire while the retired one is pushed to compute. The whole block is reserved
// up front by the caller, so chunk k lands at a fixed offset and no slot arithmetic is needed.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "vendor/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "vendor/kernel_lib/perf_instrumentation.hpp"

namespace layer_norm_tile {

// Issues `count` page reads/writes of one chunk starting at its `rotation`-th tile, so cores with
// different rotations start their bursts on different DRAM banks (bank_stagger).
template <typename Op>
FORCE_INLINE void for_each_rotated(uint32_t count, uint32_t rotation, Op op) {
    for (uint32_t step = 0; step < count; ++step) {
        uint32_t tile = step + rotation;
        tile = tile >= count ? tile - count : tile;
        op(tile);
    }
}

// One chunk of `chunk_tiles` consecutive pages between DRAM/L1 pages and contiguous local L1.
template <uint32_t chunk_tiles, uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void read_chunk(const Accessor& accessor, uint32_t first_page, uint32_t l1_addr, uint32_t rotation) {
    for_each_rotated(chunk_tiles, rotation, [&](uint32_t tile) {
        noc_async_read(accessor.get_noc_addr(first_page + tile), l1_addr + tile * tile_bytes, tile_bytes);
    });
}

template <uint32_t chunk_tiles, uint32_t tile_bytes, typename Accessor>
FORCE_INLINE void write_chunk(const Accessor& accessor, uint32_t first_page, uint32_t l1_addr, uint32_t rotation) {
    for_each_rotated(chunk_tiles, rotation, [&](uint32_t tile) {
        noc_async_write(l1_addr + tile * tile_bytes, accessor.get_noc_addr(first_page + tile), tile_bytes);
    });
}

template <uint32_t depth, typename Issue, typename Retire>
FORCE_INLINE void read_block_chunks(Noc noc, uint32_t num_chunks, Issue issue, Retire retire) {
    static_assert(depth >= 1 && depth <= NOC_MAX_TRANSACTION_ID, "trid ring depth out of the NoC id range");
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t trid = (chunk % depth) + 1;
        if (chunk >= depth) {
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid);
            retire();
        }
        dataflow_kernel_lib::set_read_trid(noc, trid);
        issue(chunk);
    }
    const uint32_t outstanding = num_chunks < depth ? num_chunks : depth;
    for (uint32_t chunk = num_chunks - outstanding; chunk < num_chunks; ++chunk) {
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, (chunk % depth) + 1);
        retire();
    }
    dataflow_kernel_lib::set_read_trid(noc, 0);
}

}  // namespace layer_norm_tile
