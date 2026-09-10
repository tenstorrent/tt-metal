// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// `load_block`'s stick read, with a core-dependent ROTATION of the issue order.
// Shared by the reader (its own row range) and the writer (the split reader's
// trailing row range) so the block operation exists once.
//
// WHAT THIS IS. Byte-for-byte the same transfers as
// `dataflow_kernel_lib::read_sticks_for_tilize<cb, TilizeGranularity::TILE>`:
// one `cb_reserve_back(cb, width_in_tiles)` / `tile_h` stick reads / ONE
// `noc_async_read_barrier()` / `cb_push_back` cycle per TILE-ROW, `row_bytes`
// per stick, source pages `start_page + blk*tile_h + row`. Same count, same
// size, same CB quantum, same barrier granularity. The ONLY difference is the
// ORDER in which the `tile_h` reads of a tile-row are issued: they start at
// `rotation % tile_h` and wrap.
//
// WHY THE REORDER IS FREE. Every stick lands at `l1_base + row * row_bytes` —
// addressed by its ROW, never by its issue slot — and all `tile_h` reads sit
// behind ONE barrier, so no consumer can observe the order. The block is
// bit-identical whatever `rotation` is, which is why `rotation` needs no
// correctness argument, only a perf one.
//
// WHY IT IS WORTH DOING (measured, Perf 1; Wormhole B0 n150, 64/64 cores).
// An interleaved tensor maps page `p` to bank `p % num_banks`
// (`tensor_accessor.h:get_bank_and_offset_from_page_id`), and `num_banks` is 12
// here. On any geometry where the cores SHARE `start_page` — every `R == 1`
// short-wide tensor, which is exactly the op's flagged perf profile
// `[1,1,32,16384]` — the ascending loop makes all 64 cores issue row 0 first,
// then row 1, then row 2, in lockstep: at read-step `r` the whole grid hammers
// bank `r % 12` while the other eleven idle. Rotating each core's start row
// de-synchronizes the grid so every bank is in service every step.
//
// Isolated reader stage on `[1,1,32,16384]` (`TILIZE_ABLATE=writes,compute`,
// NCRISC kernel span, median of 2, bit-identical throughout):
//     helper (ascending)          5545 ns mean / 7139 ns max
//     raw loop, ascending         5572 / 7409     <- the CONTROL: writing the
//                                                    loop out is worth ~0%
//     rotated (this)              4694 / 5975     <- -17.5% mean, -18.8% max
// Rotation stride is irrelevant (strides 1/5/7/11/13 and a bank-exact solve all
// land within a point of each other), so the cheapest source wins: `block_id`,
// which the block-resolve arithmetic has already computed and which is unique
// per block on every plan.
//
// RAW-API JUSTIFICATION — why `read_sticks_for_tilize` is not called here.
// CAPABILITY gap, not ergonomics: the helper's TILE-mode inner loop is
// `for (row = 0; row < rows_this_block; row++)` over `start_page + block_row +
// row` (`tilize_helpers_dataflow.inl:121`) with no order, start-offset or
// permutation parameter anywhere in its signature, so NO call-site argument can
// express a rotated start row. The measured pair above is the justification:
// the helper's own overhead is not the issue (raw-ascending ties it at 5572 vs
// 5545 ns) — only its fixed ordering is. The fix belongs IN the helper: one
// optional `start_row_offset` parameter would close this and let this file go
// back to being a call. Recorded in changelog.md's `Helper bypasses` table.
#pragma once

#include "api/dataflow/dataflow_api.h"

// Ablation switch (`TILIZE_ABLATE=reads`): drop the payload, keep the loop, the
// CB quantum and the barrier. See tilize_program_descriptor._ablation_defines.
#ifdef TILIZE_ABLATE_READS
#define TILIZE_NOC_READ(...) ((void)0)
#else
#define TILIZE_NOC_READ(...) noc_async_read(__VA_ARGS__)
#endif

namespace tilize_kernel {

// `num_tile_rows` tile-rows starting at `start_page`, `width_in_tiles` CB pages
// each. `rotation` is any core-dependent value; it is reduced mod `tile_h` and
// is a pure perf knob — 0 reproduces the helper's ascending order exactly.
template <uint32_t cb_id, uint32_t width_in_tiles, uint32_t tile_h, uint32_t row_bytes, typename Accessor>
FORCE_INLINE void read_sticks_rotated(
    const Accessor& accessor,
    uint32_t num_tile_rows,
    uint32_t start_page,
    uint32_t byte_offset_within_page,
    uint32_t rotation) {
    // `tile_h == 1` has a single stick per tile-row, so there is no order to
    // rotate and the modulo would be a divide for nothing.
    const uint32_t first_row = (tile_h > 1) ? (rotation % tile_h) : 0;
    for (uint32_t blk = 0; blk < num_tile_rows; ++blk) {
        const uint32_t page_base = start_page + blk * tile_h;
        cb_reserve_back(cb_id, width_in_tiles);
        const uint32_t l1_base = get_write_ptr(cb_id);
        // TWO STRAIGHT RUNS, not one loop with a wrap. A rotation by `first_row`
        // is the range `[first_row, tile_h)` followed by `[0, first_row)`, and
        // each run is CONTIGUOUS in both the page index and the L1 offset — so
        // both walk on plain increments, exactly like the helper's own ascending
        // loop, with no per-stick modulo, wrap branch or index-to-address
        // multiply. That matters: the first draft of this carried the wrap
        // inside the loop and measured +7.2% on `[1,1,2048,64]`, a 5 us kernel
        // where ~32 iterations of extra RISC arithmetic is a visible share of
        // the wall. The two-run form gave that back without narrowing where the
        // rotation applies — the right way to answer a regression is to delete
        // its cause, not to fence off the shape that exposed it.
        uint32_t page = page_base + first_row;
        uint32_t l1 = l1_base + first_row * row_bytes;
        for (uint32_t row = first_row; row < tile_h; ++row) {
            TILIZE_NOC_READ(accessor.get_noc_addr(page, byte_offset_within_page), l1, row_bytes);
            ++page;
            l1 += row_bytes;
        }
        page = page_base;
        l1 = l1_base;
        for (uint32_t row = 0; row < first_row; ++row) {
            TILIZE_NOC_READ(accessor.get_noc_addr(page, byte_offset_within_page), l1, row_bytes);
            ++page;
            l1 += row_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, width_in_tiles);
    }
}

}  // namespace tilize_kernel
