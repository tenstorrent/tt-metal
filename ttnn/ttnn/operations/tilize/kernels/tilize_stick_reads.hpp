// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared dataflow pieces of tilize's `load_block` / `store_block` (op_design.md):
// the per-core walk and the pipelined stick-segment producer. One source of truth
// for the traversal, so the reader, the writer and (split reader) the writer's odd
// loads cannot drift apart.
//
// Walk: a Tensix core visits its rectangle column block by column block; inside a
// column block it visits its core_row_tiles tile-rows starting at `row_rotation`
// (mod core_row_tiles). The linear position in that walk is the sequence index;
// compute consumes positions in order. Under the split reader, even positions are
// produced by NCRISC into cb_input_sticks and odd positions by BRISC into
// cb_input_sticks_odd (each CB has exactly one producer).
//
// DRAM bank spreading: with a contiguous row split every Tensix core sits at the
// same local step at the same time, so un-rotated walks send all cores to
// indices congruent modulo a small factor of the bank count (3 of 12 banks on
// Wormhole). Rotating each core's walk start (and the stick order inside a
// tile-row) by a per-core amount spreads concurrent requests over the banks.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace tilize_dataflow {

// One position of the per-core walk.
template <uint32_t block_width>
struct Walker {
    uint32_t row_start, core_row_tiles, col_start, core_col_tiles, row_rotation;
    uint32_t col_block_idx = 0;
    uint32_t i = 0;          // index within the column block's tile-row walk
    uint32_t row_local = 0;  // rotated local tile-row

    Walker(
        uint32_t row_start_,
        uint32_t core_row_tiles_,
        uint32_t col_start_,
        uint32_t core_col_tiles_,
        uint32_t rotation) :
        row_start(row_start_),
        core_row_tiles(core_row_tiles_),
        col_start(col_start_),
        core_col_tiles(core_col_tiles_),
        row_rotation(rotation % core_row_tiles_),
        row_local(rotation % core_row_tiles_) {}

    uint32_t num_positions() const { return core_row_tiles * ((core_col_tiles + block_width - 1) / block_width); }
    uint32_t row() const { return row_start + row_local; }
    uint32_t first_col() const { return col_start + col_block_idx * block_width; }
    uint32_t valid_width() const {
        const uint32_t remaining = core_col_tiles - col_block_idx * block_width;
        return remaining < block_width ? remaining : block_width;
    }
    void advance() {
        if (++row_local == core_row_tiles) {
            row_local = 0;
        }
        if (++i == core_row_tiles) {
            i = 0;
            row_local = row_rotation;
            ++col_block_idx;
        }
    }
};

// Pipelined producer of tile-row slots in one input CB.
//
// issue(): reserve room for (outstanding + 1) tile-rows, then issue the tile_h
//          stick-segment reads of one tile-row into the next slot under that slot's
//          NoC transaction id. If `read_ahead` tile-rows are already outstanding,
//          the oldest is completed first.
// complete_oldest(): barrier on the oldest slot's transaction id, push block_width.
//
// Every push is the nominal block_width pages, so the CB write pointer after n
// pushes is base + (n % depth) * slot_bytes; slot addresses are derived from that
// instead of CB internals. Requires read_ahead <= depth and depth <= 15 (trids).
template <
    uint32_t cb,
    uint32_t block_width,
    uint32_t depth,
    uint32_t read_ahead,
    uint32_t tile_h,
    uint32_t tile_col_bytes,
    uint32_t in_tile_bytes>
struct StickProducer {
    static_assert((tile_h & (tile_h - 1)) == 0, "tile_h must be a power of two");
    static_assert(read_ahead >= 1 && read_ahead <= depth, "read_ahead must be in [1, depth]");
    static_assert(depth <= 15, "one NoC transaction id per CB slot");
    static constexpr uint32_t slot_bytes = block_width * in_tile_bytes;
    static constexpr uint32_t block_stick_bytes = block_width * tile_col_bytes;  // nominal L1 stride per stick

    uint32_t base_addr;
    uint32_t stick_rotation;
    uint32_t next_slot = 0;    // slot index of the next issue (monotonic)
    uint32_t outstanding = 0;  // issued, not yet pushed

    explicit StickProducer(uint32_t rotation) : base_addr(get_write_ptr(cb)), stick_rotation(rotation & (tile_h - 1)) {}

    static uint32_t trid_of(uint32_t slot) { return 1 + (slot % depth); }

    template <typename Accessor>
    FORCE_INLINE void issue(const Accessor& accessor, uint32_t row, uint32_t first_col, uint32_t valid_width) {
        if (outstanding == read_ahead) {
            complete_oldest();
        }
        cb_reserve_back(cb, block_width * (outstanding + 1));
        const uint32_t l1_base = base_addr + (next_slot % depth) * slot_bytes;
        const uint32_t segment_bytes = valid_width * tile_col_bytes;
        const uint32_t segment_offset = first_col * tile_col_bytes;
        const uint32_t first_stick = row * tile_h;
        noc_async_read_set_trid(trid_of(next_slot));
        for (uint32_t s = 0; s < tile_h; ++s) {
            const uint32_t stick = (s + stick_rotation) & (tile_h - 1);
            noc_async_read(
                accessor.get_noc_addr(first_stick + stick, segment_offset),
                l1_base + stick * block_stick_bytes,
                segment_bytes);
        }
        ++next_slot;
        ++outstanding;
    }

    FORCE_INLINE void complete_oldest() {
        const uint32_t oldest = next_slot - outstanding;
        noc_async_read_barrier_with_trid(trid_of(oldest));
        cb_push_back(cb, block_width);
        --outstanding;
    }

    FORCE_INLINE void complete_all() {
        while (outstanding > 0) {
            complete_oldest();
        }
        noc_async_read_set_trid(0);
    }
};

// store_block for one tile-row: wait block_width pages, valid_width tile-page writes
// in flight, one flush (L1 source reads done), pop block_width.
template <uint32_t cb_output_tiles, uint32_t block_width, uint32_t out_tile_bytes, typename Accessor>
FORCE_INLINE void store_tile_row(const Accessor& accessor, uint32_t first_tile_idx, uint32_t valid_width) {
    cb_wait_front(cb_output_tiles, block_width);
    uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
    uint32_t tile_idx = first_tile_idx;
    for (uint32_t t = 0; t < valid_width; ++t) {
        noc_async_write(l1_read_addr, accessor.get_noc_addr(tile_idx), out_tile_bytes);
        l1_read_addr += out_tile_bytes;
        ++tile_idx;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_output_tiles, block_width);
}

}  // namespace tilize_dataflow
