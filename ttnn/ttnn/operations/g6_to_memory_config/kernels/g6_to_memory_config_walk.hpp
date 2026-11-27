// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the gather (reader) and scatter (writer) kernels of g6_to_memory_config.
//
// A mover walks a sequence of tile pages in a fixed order and moves each maximal run that is
// contiguous on both sides in one NoC transaction (a segment). Page id = tile_row * tensor_cols + tile_col
// under every placement.
//
// Compile-time args (identical layout in both kernels):
//   0 local_kind           LocalKind of the L1 side this core owns
//   1 walk_sharded         1: walk shards of the walk placement in slot order; 0: walk page ids linearly
//   2 tensor_pages
//   3 tensor_rows_tiles    leading dims folded, each image tile-padded on its own
//   4 tensor_cols_tiles
//   5 walk_shard_rows      shard extents (tiles) of the placement that orders the walk
//   6 walk_shard_cols
//   7 walk_num_shard_cols  shards per shard-grid row
//   8 max_segment_pages    cap of one NoC transaction
//   9 block_pages          positions per block
//  10 cb_staging_pages     staging CB index (bounce only)
//  11.. TensorAccessorArgs(source), then TensorAccessorArgs(target)
// Common runtime args: source base address, target base address.
// Per-core runtime args: walk start (shard id or page id), position offset inside that shard, position count.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6 {

enum LocalKind : uint32_t {
    OWNED_SHARD = 0,  // this core holds one shard of the local placement
    OWNED_BANKS = 1,  // this core hosts L1 bank(s) of an interleaved local placement
    STAGING = 2,      // no side is local: pages rest in the staging CB (bounce)
};

constexpr uint32_t UNBOUNDED = 0xFFFFFFFFu;

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Shards in shard-id order (row-major over the shard grid), each walked row-major over its real pages.
// A shard's slot stride is the full shard width, also in a ragged last shard.
struct ShardWalk {
    static constexpr bool is_sharded = true;
    uint32_t tensor_rows, tensor_cols, shard_rows, shard_cols, num_shard_cols;
    uint32_t shard_id = 0, row0 = 0, col0 = 0, rows = 0, cols = 1, r = 0, c = 0;

    void enter(uint32_t id, uint32_t offset) {
        shard_id = id;
        row0 = (id / num_shard_cols) * shard_rows;
        col0 = (id % num_shard_cols) * shard_cols;
        if (row0 >= tensor_rows) {
            return;  // past the last shard: only reached after the final position
        }
        rows = min_u32(shard_rows, tensor_rows - row0);
        cols = min_u32(shard_cols, tensor_cols - col0);
        r = offset / cols;
        c = offset % cols;
    }
    uint32_t page() const { return (row0 + r) * tensor_cols + col0 + c; }
    uint32_t slot() const { return r * shard_cols + c; }
    // Positions ahead (this one included) whose page ids step by 1.
    uint32_t page_run() const { return cols == tensor_cols ? (rows - r) * cols - c : cols - c; }
    // Positions ahead whose shard slots step by 1.
    uint32_t slot_run() const { return cols == shard_cols ? (rows - r) * cols - c : cols - c; }
    void advance(uint32_t n) {
        const uint32_t k = c + n;
        r += k / cols;
        c = k % cols;
        if (r >= rows) {
            enter(shard_id + 1, 0);
        }
    }
};

// Page ids start, start + stride, start + 2 * stride, ...
struct LinearWalk {
    static constexpr bool is_sharded = false;
    uint32_t page_id = 0, stride = 1;
    uint32_t page() const { return page_id; }
    uint32_t slot() const { return 0; }
    uint32_t page_run() const { return UNBOUNDED; }
    uint32_t slot_run() const { return UNBOUNDED; }
    void advance(uint32_t n) { page_id += n * stride; }
};

// Pages from `page` (walk step `step`, at most `limit` of them) that sit back to back in the remote buffer.
template <typename Accessor>
FORCE_INLINE uint32_t
remote_run(const Accessor& remote, uint32_t remote_stride, uint32_t page, uint32_t step, uint32_t limit) {
    if (limit <= 1 || remote_stride != step) {
        return 1;
    }
    return remote.num_contiguous_pages(page, page + (limit - 1) * step + 1);
}

// Moves `count` positions of `walk` between the remote buffer and local L1, one NoC call per segment.
// IS_READ: remote -> local (gather); otherwise local -> remote (scatter). No barrier inside.
// Local address: OWNED_SHARD -> local_base + slot * page; otherwise local_base advances with the walk.
template <bool IS_READ, uint32_t LOCAL_KIND, bool REMOTE_DEFINES_WALK, typename Walk, typename Accessor>
FORCE_INLINE void move_positions(
    Walk& walk,
    uint32_t step,
    const Accessor& remote,
    uint32_t remote_stride,
    uint32_t local_base,
    uint32_t page_size,
    uint32_t max_segment_pages,
    uint32_t count) {
    uint32_t local_addr = local_base;
    while (count > 0) {
        uint32_t limit = min_u32(count, max_segment_pages);
        if constexpr (LOCAL_KIND == OWNED_SHARD) {
            limit = min_u32(limit, walk.slot_run());
            local_addr = local_base + walk.slot() * page_size;
        }
        const uint32_t page = walk.page();
        uint32_t n;
        if constexpr (REMOTE_DEFINES_WALK && Walk::is_sharded) {
            n = min_u32(limit, walk.slot_run());
        } else {
            n = remote_run(remote, remote_stride, page, step, min_u32(limit, walk.page_run()));
        }
        const uint64_t remote_addr = remote.get_noc_addr(page);
        const uint32_t bytes = n * page_size;
        if constexpr (IS_READ) {
            noc_async_read(remote_addr, local_addr, bytes);
        } else {
            noc_async_write(local_addr, remote_addr, bytes);
        }
        local_addr += bytes;
        walk.advance(n);
        count -= n;
    }
}

// R1 pull / R2 push over the owned set, `block_pages` positions per block. No barrier: the caller drains once.
template <bool IS_READ, uint32_t LOCAL_KIND, typename Walk, typename Accessor>
FORCE_INLINE void move_owned(
    Walk& walk,
    uint32_t step,
    const Accessor& remote,
    uint32_t remote_stride,
    uint32_t local_base,
    uint32_t page_size,
    uint32_t max_segment_pages,
    uint32_t block_pages,
    uint32_t count) {
    for (uint32_t done = 0; done < count; done += block_pages) {
        const uint32_t n = min_u32(block_pages, count - done);
        const uint32_t block_base = LOCAL_KIND == OWNED_SHARD ? local_base : local_base + done * page_size;
        move_positions<IS_READ, LOCAL_KIND, false>(
            walk, step, remote, remote_stride, block_base, page_size, max_segment_pages, n);
    }
}

// L1 banks of an interleaved buffer hosted by this core, found through the firmware bank table.
// One bank per worker core when the host's l1_bank_owners predicate holds.
constexpr uint32_t MAX_OWNED_BANKS = 4;

FORCE_INLINE uint32_t find_owned_l1_banks(uint32_t tensor_pages, uint32_t (&banks)[MAX_OWNED_BANKS]) {
    uint32_t num_owned = 0;
    for (uint32_t bank = 0; bank < NUM_L1_BANKS && bank < tensor_pages; ++bank) {
        if (noc_address_backend::is_local(noc_address_backend::bank_address<false>(bank, 0, noc_index), noc_index)) {
            ASSERT(num_owned < MAX_OWNED_BANKS);
            banks[num_owned++] = bank;
        }
    }
    return num_owned;
}

// Pages b, b + NUM_L1_BANKS, ... below tensor_pages.
FORCE_INLINE uint32_t bank_page_count(uint32_t bank, uint32_t tensor_pages) {
    return (tensor_pages - bank + NUM_L1_BANKS - 1) / NUM_L1_BANKS;
}

}  // namespace g6
