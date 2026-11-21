// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// split reader: fills cb_output_pages with the output pages of this core's contiguous range of the walk.
//
// regime page_remap (implicit sync): each unit is one input page, which is exactly one output page; one tagged
//   read per entry lands at the ring's cursor and the DM0 interrupt handler posts its credit.
// regime tile_row_shift (explicit sync on cb_output_pages, opted out on the host): the cut is inside a tile, so an
//   output tile takes its rows from up to two input tiles. Per block of <= block_h_tiles output tiles of one
//   tile column:
//     load_window_block  read the input tiles the block needs that are not yet resident into the scratchpad
//                        window (slot = tile-row % window_tiles), ONE barrier for the block;
//     assemble_block     per output tile: reserve 1 entry, copy its valid rows from the window with <= 8
//                        self-aimed NoC copies (face-row runs), barrier, push 1.
//   Every transfer of this regime carries the user transaction id 0 (target page, Ring protocols).
//
// No kernel-library helper covers a DRAM page -> ring read or the row-run assembly. The self-aimed copies use the
// Scratchpad itself as the NoC source (a typed loopback endpoint, internal/local_mem/noc_traits.h) instead of
// dataflow_kernel_lib::local_addr(raw address): same transfer, with the scratchpad's bounds check.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "split_walk.hpp"

namespace {

// Byte offset of row `r`, column half `h` (0: columns 0-15, 1: columns 16-31) inside a 32x32 tile of four
// 16x16 faces stored face after face.
inline uint32_t tile_row_offset(uint32_t r, uint32_t h, uint32_t face_bytes, uint32_t face_row_bytes) {
    return ((r / 16) * 2 + h) * face_bytes + (r % 16) * face_row_bytes;
}

}  // namespace

void kernel_main() {
    const uint32_t start_unit = get_arg(args::start_unit);
    const uint32_t num_units = get_arg(args::num_units);
    constexpr uint32_t R = get_arg(args::reader_threads);

    DataflowBuffer ring(dfb::cb_output_pages);
    Noc noc;
    const auto acc_input = TensorAccessor(tensor::input);

#if SPLIT_REGIME == SPLIT_REGIME_PAGE_REMAP
    {
        split::PageRemapWalk walk(get_arg(args::cut_pages), get_arg(args::inner_pages), get_arg(args::chunk_pages));
        // Hart t of R takes units t, t+R, ... of the range (the ring deals entries to the writer harts alike).
        walk.seek(start_unit + get_my_thread_id());
        for (uint32_t n = get_my_thread_id(); n < num_units; n += R) {
            noc.async_read<NocOptions::TXN_ID>(acc_input, ring, {.page_id = walk.input_page()}, {});
            walk.advance(R);
        }
        // Firmware credits a last partial batch when the kernel returns. Hand the hart back with the user id in
        // its sticky tag register, so a following program's untagged transfers are not counted for a ring.
        noc_async_read_set_trid(0, noc.get_noc_id());
    }
#elif SPLIT_REGIME == SPLIT_REGIME_TILE_ROW_SHIFT
    {
        static_assert(R == 1, "the tile_row_shift reader is one hart: its window is per hart");
        constexpr uint32_t block_h_tiles = get_arg(args::block_h_tiles);
        constexpr uint32_t window_tiles = get_arg(args::window_tiles);
        static_assert(window_tiles >= block_h_tiles + 1, "a block spans up to block_h_tiles + 1 input tile-rows");
        constexpr NocOptVals user_txn = {.trid = 0};

        Scratchpad<uint8_t> window(scratch::scratch_input_window);
        const uint32_t page_bytes = ring.get_entry_size();
        const uint32_t face_bytes = page_bytes / 4;
        const uint32_t face_row_bytes = page_bytes / 64;
        auto slot_offset = [&](uint32_t tile_row) { return (tile_row % window_tiles) * page_bytes; };

        split::TileRowShiftWalk walk(
            get_arg(args::h_tiles),
            get_arg(args::w_tiles),
            get_arg(args::split_rows),
            get_arg(args::cut_rows),
            get_arg(args::chunk_tiles),
            get_arg(args::column_tiles));
        walk.seek(start_unit);

        // Resident input tile-rows [lo, hi) of column `resident_col` (none at the start of the range).
        uint32_t resident_col = 0xFFFFFFFFu, lo = 0, hi = 0;
        uint32_t remaining = num_units;
        while (remaining > 0) {
            const uint32_t block = split::min_u32(split::min_u32(block_h_tiles, remaining), walk.tiles_left_in_column());

            // ---- load_window_block: the block's input rows are contiguous: [first row, end of last tile's rows)
            const uint32_t first_row = walk.first_row();
            split::TileRowShiftWalk last = walk;
            last.advance(block - 1);
            const uint32_t end_row = last.first_row() + last.valid_rows();
            const uint32_t ta_lo = first_row / split::TILE_ROWS;
            const uint32_t ta_hi = (end_row - 1) / split::TILE_ROWS + 1;
            if (walk.col != resident_col || ta_lo >= hi || ta_lo < lo) {
                resident_col = walk.col;
                lo = hi = ta_lo;
            }
            lo = ta_lo;  // tiles below the block are no longer needed: their slots may be refilled
            if (ta_hi > hi) {
                for (uint32_t ta = hi; ta < ta_hi; ++ta) {
                    noc.async_read<NocOptions::TXN_ID>(
                        acc_input, window, page_bytes, {.page_id = walk.input_page(ta)}, {.offset_bytes = slot_offset(ta)}, user_txn);
                }
                hi = ta_hi;
                noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
            }

            // ---- assemble_block: one ring entry per output tile
            for (uint32_t b = 0; b < block; ++b) {
                const uint32_t g = walk.first_row();
                const uint32_t ta = g / split::TILE_ROWS;
                const uint32_t shift = g % split::TILE_ROWS;
                const uint32_t valid = walk.valid_rows();
                ring.reserve_back(1);
                if (shift == 0) {
                    // Rows line up with the input tile: copy it whole (rows past `valid` are padding).
                    noc.async_read<NocOptions::TXN_ID>(
                        window, ring, page_bytes, {.offset_bytes = slot_offset(ta)}, {.offset_bytes = 0}, user_txn);
                } else {
                    // Output row r <- input row r + shift (of tile ta, or of tile ta + 1 past row 31). Maximal runs
                    // that stay inside one 16-row face band on both sides; each run is one copy per column half.
                    uint32_t r = 0;
                    while (r < valid) {
                        const uint32_t s = r + shift;
                        const uint32_t src_tile = ta + s / split::TILE_ROWS;
                        const uint32_t sr = s % split::TILE_ROWS;
                        const uint32_t run = split::min_u32(split::min_u32(valid - r, 16 - r % 16), 16 - sr % 16);
                        const uint32_t run_bytes = run * face_row_bytes;
                        for (uint32_t h = 0; h < 2; ++h) {
                            noc.async_read<NocOptions::TXN_ID>(
                                window,
                                ring,
                                run_bytes,
                                {.offset_bytes = slot_offset(src_tile) + tile_row_offset(sr, h, face_bytes, face_row_bytes)},
                                {.offset_bytes = tile_row_offset(r, h, face_bytes, face_row_bytes)},
                                user_txn);
                        }
                        r += run;
                    }
                }
                noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                ring.push_back(1);
                walk.next();
            }
            remaining -= block;
        }
        noc_async_read_set_trid(0, noc.get_noc_id());
    }
#else
#error "split_reader: unknown SPLIT_REGIME"
#endif
}
