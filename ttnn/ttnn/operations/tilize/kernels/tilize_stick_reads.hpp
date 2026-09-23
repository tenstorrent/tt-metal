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

// Pipelined producer of CB slots in one input CB. A slot holds `rows_per_slot`
// consecutive walk positions (tile-rows of block_width pages each), so the read
// barrier and the CB push happen once per slot, not once per tile-row
// (`rows_per_slot` = the host-derived rows_per_quantum, see QUANTUM_MIN_TILES).
//
// issue_row(): opening a slot reserves room for (outstanding + 1) slots and tags
//              the slot's reads with its NoC transaction id; if `read_ahead` slots
//              are already outstanding, the oldest is completed first. Then the
//              tile_h stick-segment reads of one tile-row land in the open slot.
// complete_oldest(): barrier on the oldest slot's transaction id, push its pages.
//
// Every push is the nominal rows_per_slot * block_width pages except the very
// last one of the kernel (a partial final slot), so the CB write pointer after
// n full pushes is base + (n % depth) * slot_bytes; slot addresses are derived
// from that instead of CB internals. Nothing is pushed after the partial slot,
// so it cannot break the ring-wrap invariant. Requires read_ahead <= depth and
// depth <= 15 (one trid per slot).
//
// Input pages: interleaved and HEIGHT-sharded Layout::ROW_MAJOR tensors keep one
// page per stick (pages_per_stick == 1: one NoC read per stick segment). A
// WIDTH / BLOCK / ND-sharded one cuts each stick into pages of the shard width
// (`page_bytes` data bytes each; page id = stick * pages_per_stick + k), so a
// segment is read as one NoC read per page it overlaps (read_paged_segment).
template <
    uint32_t cb,
    uint32_t block_width,
    uint32_t depth,
    uint32_t read_ahead,
    uint32_t tile_h,
    uint32_t tile_col_bytes,
    uint32_t rows_per_slot,
    uint32_t page_bytes,
    uint32_t pages_per_stick>
struct StickProducer {
    static_assert((tile_h & (tile_h - 1)) == 0, "tile_h must be a power of two");
    static_assert(read_ahead >= 1 && read_ahead <= depth, "read_ahead must be in [1, depth]");
    static_assert(depth <= 15, "one NoC transaction id per CB slot");
    static_assert(rows_per_slot >= 1, "a slot holds at least one tile-row");
    static constexpr uint32_t in_tile_bytes = tile_h * tile_col_bytes;  // tile-sized page: tile_h stick segments
    static constexpr uint32_t row_pages = block_width;                  // pages per tile-row
    static constexpr uint32_t row_bytes = block_width * in_tile_bytes;
    static constexpr uint32_t slot_pages = rows_per_slot * row_pages;
    static constexpr uint32_t slot_bytes = rows_per_slot * row_bytes;
    static constexpr uint32_t block_stick_bytes = block_width * tile_col_bytes;  // nominal L1 stride per stick

    uint32_t base_addr;
    uint32_t stick_rotation;
    uint32_t next_slot = 0;    // slot index of the next slot to open (monotonic)
    uint32_t outstanding = 0;  // opened, not yet pushed
    uint32_t open_rows = 0;    // tile-rows already issued into the newest slot (0 = no slot open)
    uint32_t open_base = 0;    // L1 address of the newest slot

    explicit StickProducer(uint32_t rotation) : base_addr(get_write_ptr(cb)), stick_rotation(rotation & (tile_h - 1)) {}

    static uint32_t trid_of(uint32_t slot) { return 1 + (slot % depth); }

    // Tile-rows fully published (pushed) so far.
    uint32_t rows_pushed() const { return (next_slot - outstanding) * rows_per_slot; }

    template <typename Accessor>
    FORCE_INLINE void issue_row(const Accessor& accessor, uint32_t row, uint32_t first_col, uint32_t valid_width) {
        if (open_rows == 0) {
            if (outstanding == read_ahead) {
                complete_oldest();
            }
            cb_reserve_back(cb, slot_pages * (outstanding + 1));
            open_base = base_addr + (next_slot % depth) * slot_bytes;
            noc_async_read_set_trid(trid_of(next_slot));
            ++next_slot;
            ++outstanding;
        }
        const uint32_t l1_base = open_base + open_rows * row_bytes;
        const uint32_t segment_bytes = valid_width * tile_col_bytes;
        const uint32_t segment_offset = first_col * tile_col_bytes;
        const uint32_t first_stick = row * tile_h;
        for (uint32_t s = 0; s < tile_h; ++s) {
            const uint32_t stick = (s + stick_rotation) & (tile_h - 1);
            const uint32_t l1_dst = l1_base + stick * block_stick_bytes;
            if constexpr (pages_per_stick == 1) {
                noc_async_read(accessor.get_noc_addr(first_stick + stick, segment_offset), l1_dst, segment_bytes);
            } else {
                read_paged_segment(accessor, first_stick + stick, segment_offset, l1_dst, segment_bytes);
            }
        }
        if (++open_rows == rows_per_slot) {
            open_rows = 0;  // slot sealed; it completes lazily
        }
    }

    // Bytes [offset, offset + bytes) of logical stick `stick`, split at page boundaries.
    template <typename Accessor>
    FORCE_INLINE static void read_paged_segment(
        const Accessor& accessor, uint32_t stick, uint32_t offset, uint32_t l1_dst, uint32_t bytes) {
        uint32_t page = stick * pages_per_stick + offset / page_bytes;
        uint32_t in_page = offset % page_bytes;
        while (bytes > 0) {
            const uint32_t chunk = bytes < page_bytes - in_page ? bytes : page_bytes - in_page;
            noc_async_read(accessor.get_noc_addr(page, in_page), l1_dst, chunk);
            l1_dst += chunk;
            bytes -= chunk;
            ++page;
            in_page = 0;
        }
    }

    FORCE_INLINE void complete_oldest() {
        const uint32_t oldest = next_slot - outstanding;
        // Only the newest slot can be partial, and only at kernel end (complete_all).
        const bool partial = outstanding == 1 && open_rows != 0;
        noc_async_read_barrier_with_trid(trid_of(oldest));
        cb_push_back(cb, partial ? open_rows * row_pages : slot_pages);
        if (partial) {
            open_rows = 0;
        }
        --outstanding;
    }

    FORCE_INLINE void complete_all() {
        while (outstanding > 0) {
            complete_oldest();
        }
        noc_async_read_set_trid(0);
    }
};

// store_block for `num_rows` consecutive walk positions (one CB quantum): wait
// num_rows * block_width pages, valid_width tile-page writes per tile-row in
// flight, one flush (L1 source reads done), pop. `walk` is advanced past them.
// The writes of one tile-row start at column `col_rotation % valid_width` and
// wrap (the write twin of the reader's stick rotation): when every Tensix core
// holds the same few tile-rows at once, concurrent first writes would otherwise
// all land on the banks of column 0.
template <uint32_t cb_output_tiles, uint32_t block_width, uint32_t out_tile_bytes, typename Accessor, typename Walk>
FORCE_INLINE void store_rows(
    const Accessor& accessor, Walk& walk, uint32_t tiles_per_row, uint32_t num_rows, uint32_t col_rotation) {
    const uint32_t pages = num_rows * block_width;
    cb_wait_front(cb_output_tiles, pages);
    uint32_t l1_row_addr = get_read_ptr(cb_output_tiles);
    for (uint32_t j = 0; j < num_rows; ++j, walk.advance()) {
        const uint32_t row_tile_idx = walk.row() * tiles_per_row + walk.first_col();
        const uint32_t valid_width = walk.valid_width();
        uint32_t t = col_rotation % valid_width;
        for (uint32_t n = 0; n < valid_width; ++n) {
            noc_async_write(l1_row_addr + t * out_tile_bytes, accessor.get_noc_addr(row_tile_idx + t), out_tile_bytes);
            if (++t == valid_width) {
                t = 0;
            }
        }
        l1_row_addr += block_width * out_tile_bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_output_tiles, pages);
}

}  // namespace tilize_dataflow
