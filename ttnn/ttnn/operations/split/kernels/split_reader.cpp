// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// split reader: fills cb_output_pages with the output pages of this core's contiguous range of the walk.
//
// regime page_remap (implicit sync): each unit is one input page, which is exactly one output page; one tagged
//   read per entry lands at the ring's cursor and the DM0 interrupt handler posts its credit.
// shift regimes (explicit sync on cb_output_pages, opted out on the host): an output page takes its elements from
//   up to two input pages of one line of the cut axis (split_walk.hpp, ShiftWalk):
//     tile_row_shift  TILE, dim -2, cut inside a tile: rows of up to two vertically adjacent input tiles;
//     tile_col_shift  TILE, dim -1, cut inside a tile: columns of up to two horizontally adjacent input tiles;
//     rm_substick     ROW_MAJOR, dim -1: an element range of one input stick.
//   Per block of <= block_h_tiles output pages of the walk (a block may span several lines):
//     load_window_block  read the input pages the block needs that are not yet resident into the scratchpad
//                        window (slot = read sequence number % window_tiles), ONE barrier for the block;
//     assemble_block     per output page: reserve 1 entry, assemble it from the window, push 1.
//                        tile_row_shift: <= 8 self-aimed NoC copies of face-row runs, barrier.
//                        tile_col_shift: shift 0 -> one whole-tile self-aimed copy; else RISC-V copies of <= 4
//                          column runs per row (column shifts are element-granular: below the 16-B L1 alignment a
//                          NoC copy cannot realign them).
//                        rm_substick: one self-aimed copy when the range starts 16-B aligned, else RISC-V copies.
//   RISC-V reads of the window go through its scoped_lock (cache invalidated after the block's barrier); RISC-V
//   writes of the entry through the ring's scoped_write_lock (uncached alias, flushed on release, before push).
//   Every NoC transfer of these regimes carries the user transaction id 0 (target page, Ring protocols).
//
// No kernel-library helper covers a DRAM page -> ring read or the row-run assembly. The self-aimed copies use the
// Scratchpad itself as the NoC source (a typed loopback endpoint, internal/local_mem/noc_traits.h) instead of
// dataflow_kernel_lib::local_addr(raw address): same transfer, with the scratchpad's bounds check.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/core_local_mem.h"
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


// Copy `bytes` bytes (a whole number of elements, >= 2 B each) from `src` to `dst`, in the widest words both
// addresses and the length allow (4 B when all three are 4-B multiples, else 2 B).
inline void copy_elems(CoreLocalMem<uint8_t> dst, CoreLocalMem<uint8_t> src, uint32_t bytes) {
    volatile tt_l1_ptr uint8_t* d = &dst[0];
    const volatile tt_l1_ptr uint8_t* s = &src[0];
    if (((reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(s) | bytes) & 3u) == 0) {
        auto* d4 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d);
        auto* s4 = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(s);
        for (uint32_t n = 0; n < bytes / 4; ++n) {
            d4[n] = s4[n];
        }
    } else {
        auto* d2 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(d);
        auto* s2 = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(s);
        for (uint32_t n = 0; n < bytes / 2; ++n) {
            d2[n] = s2[n];
        }
    }
}

// Element index of (row r, column q) inside a 32x32 tile of four 16x16 faces stored face after face.
inline uint32_t tile_elem(uint32_t r, uint32_t q) { return ((r / 16) * 2 + q / 16) * 256 + (r % 16) * 16 + q % 16; }

// tile_col_shift assembly: output column q in [0, valid) <- input column q + shift of tile `src0`, or column
// q + shift - 32 of tile `src1` past column 31, for rows [0, rows). The column runs (split where either side crosses
// a 16-column face boundary or the input crosses into src1) are the same for every row: <= 4 runs, computed once.
inline void assemble_col_shift(
    CoreLocalMem<uint8_t> dst,
    CoreLocalMem<uint8_t> src0,
    CoreLocalMem<uint8_t> src1,
    uint32_t shift,
    uint32_t valid,
    uint32_t rows,
    uint32_t elem_bytes) {
    struct Run {
        uint32_t dq, sq, len;
        bool second;
    };
    Run runs[4];
    uint32_t n_runs = 0;
    for (uint32_t q = 0; q < valid;) {
        const uint32_t s = q + shift;  // column in the 64-column [src0 | src1] pair
        const uint32_t len = split::min_u32(split::min_u32(valid - q, 16 - q % 16), 16 - s % 16);
        runs[n_runs++] = Run{q, s % 32, len, s >= 32};
        q += len;
    }
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < n_runs; ++k) {
            const Run& run = runs[k];
            copy_elems(
                dst + tile_elem(r, run.dq) * elem_bytes,
                (run.second ? src1 : src0) + tile_elem(r, run.sq) * elem_bytes,
                run.len * elem_bytes);
        }
    }
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
        // its sticky tag register.
        noc_async_read_set_trid(0, noc.get_noc_id());
    }
#elif SPLIT_REGIME_IS_SHIFT
    {
        static_assert(R == 1, "the shift-regime reader is one hart: its window is per hart");
        constexpr uint32_t block_units = get_arg(args::block_h_tiles);
        constexpr uint32_t window_tiles = get_arg(args::window_tiles);
        constexpr uint32_t l1_alignment = get_arg(args::l1_alignment);
        // A block of block_units output pages may span several lines (a short line holds one or two output pages):
        // a line segment of m output pages reads at most m + 1 input pages, so a block reads at most
        // 2 * block_units, the one it may share with the previous block included.
        static_assert(window_tiles >= 2 * block_units, "a block's input pages must fit the window");
        constexpr NocOptVals user_txn = {.trid = 0};

        Scratchpad<uint8_t> window(scratch::scratch_input_window);
        // Window slots hold input pages; ring entries output pages (the same size, except rm_substick).
        const uint32_t in_page_bytes = get_arg(args::in_page_bytes);
        const uint32_t elem_bytes = get_arg(args::elem_bytes);
        // Input pages are numbered in the order they are read (seq); a page's slot is seq % window_tiles. The walk
        // needs input pages (line, t) in increasing order, so a block's pages have consecutive seqs, and the last
        // page read is the only resident page a later output page can need again.
        auto slot_offset = [&](uint32_t seq) { return (seq % window_tiles) * in_page_bytes; };

        split::ShiftWalk walk(
            get_arg(args::cut_pages),
            get_arg(args::inner_pages),
            get_arg(args::split_elems),
            get_arg(args::cut_elems),
            get_arg(args::unit_elems),
            get_arg(args::page_elems),
            get_arg(args::chunk_units),
            get_arg(args::line_units));
        walk.seek(start_unit);

#if SPLIT_REGIME == SPLIT_REGIME_TILE_COL_SHIFT
        // Valid tile rows of a tile row th: rows past the input's H are padding, not copied.
        const uint32_t cross_elems = get_arg(args::cross_elems);
        const uint32_t cross_pages = get_arg(args::cross_pages);
#endif

        uint32_t last_line = 0xFFFFFFFFu, last_t = 0, last_seq = 0;  // the last input page read (none yet)
        uint32_t next_seq = 0;
        uint32_t first_seq[block_units];  // seq of each output page's first input page; the second is seq + 1
        uint32_t remaining = num_units;
        while (remaining > 0) {
            const uint32_t block = split::min_u32(block_units, remaining);

            // ---- load_window_block: every input page the block needs that is not resident, ONE barrier
            const uint32_t block_first_seq = next_seq;
            split::ShiftWalk w = walk;
            for (uint32_t b = 0; b < block; ++b, w.next()) {
                const uint32_t ta = w.ta(), tb = w.tb();
                for (uint32_t t = ta; t <= tb; ++t) {
                    if (w.line == last_line && t == last_t) {
                        continue;  // resident
                    }
                    last_line = w.line, last_t = t, last_seq = next_seq++;
                    noc.async_read<NocOptions::TXN_ID>(
                        acc_input,
                        window,
                        in_page_bytes,
                        {.page_id = w.input_page(t)},
                        {.offset_bytes = slot_offset(last_seq)},
                        user_txn);
                }
                first_seq[b] = tb == ta ? last_seq : last_seq - 1;
            }
            if (next_seq != block_first_seq) {
                noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
#if SPLIT_REGIME != SPLIT_REGIME_TILE_ROW_SHIFT
                // The CPU assembly below reads the freshly landed pages: drop any stale cached copy of their slots
                // (the lock's acquire invalidates the range; nothing is written through it).
                for (uint32_t seq = block_first_seq; seq != next_seq; ++seq) {
                    [[maybe_unused]] auto fresh = window.scoped_lock(slot_offset(seq), in_page_bytes);
                }
#endif
            }

            // ---- assemble_block: one ring entry per output page
            for (uint32_t b = 0; b < block; ++b) {
                const uint32_t shift = walk.shift();
                const uint32_t valid = walk.valid();
                ring.reserve_back(1);
#if SPLIT_REGIME == SPLIT_REGIME_RM_SUBSTICK
                // Output stick = bytes [shift, shift + valid) * elem of the input stick.
                const uint32_t src_off = slot_offset(first_seq[b]) + shift * elem_bytes;
                const uint32_t valid_bytes = valid * elem_bytes;
                if (src_off % l1_alignment == 0) {
                    // Aligned: one self-aimed copy. Rounding up stays inside the input slot and the entry (both are
                    // aligned pages at least as long as the rounded range).
                    noc.async_read<NocOptions::TXN_ID>(
                        window,
                        ring,
                        (valid_bytes + l1_alignment - 1) / l1_alignment * l1_alignment,
                        {.offset_bytes = src_off},
                        {.offset_bytes = 0},
                        user_txn);
                    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                } else {
                    // Element-granular realignment (no NoC copy can move an L1 range to a different 16-B phase).
                    auto entry = ring.scoped_write_lock(1);
                    copy_elems(entry.get_ptr<uint8_t>(), window.local_mem() + src_off, valid_bytes);
                }
#elif SPLIT_REGIME == SPLIT_REGIME_TILE_COL_SHIFT
                if (shift == 0) {
                    // Columns line up with the input tile: copy it whole (columns past `valid` are padding).
                    noc.async_read<NocOptions::TXN_ID>(
                        window, ring, in_page_bytes, {.offset_bytes = slot_offset(first_seq[b])}, {.offset_bytes = 0}, user_txn);
                    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                } else {
                    const uint32_t th = (walk.line / walk.inner_pages) % cross_pages;
                    const uint32_t rows = split::min_u32(32, cross_elems - 32 * th);
                    auto entry = ring.scoped_write_lock(1);
                    assemble_col_shift(
                        entry.get_ptr<uint8_t>(),
                        window.local_mem() + slot_offset(first_seq[b]),
                        window.local_mem() + slot_offset(first_seq[b] + 1),
                        shift,
                        valid,
                        rows,
                        elem_bytes);
                }
#else  // SPLIT_REGIME_TILE_ROW_SHIFT
                {
                    const uint32_t face_bytes = in_page_bytes / 4;
                    const uint32_t face_row_bytes = in_page_bytes / 64;
                    if (shift == 0) {
                        // Rows line up with the input tile: copy it whole (rows past `valid` are padding).
                        noc.async_read<NocOptions::TXN_ID>(
                            window, ring, in_page_bytes, {.offset_bytes = slot_offset(first_seq[b])}, {.offset_bytes = 0}, user_txn);
                    } else {
                        // Output row r <- input row r + shift (of tile ta, or of tile ta + 1 past row 31). Maximal
                        // runs that stay inside one 16-row face band on both sides; each run is one copy per column
                        // half.
                        uint32_t r = 0;
                        while (r < valid) {
                            const uint32_t s = r + shift;
                            const uint32_t src_seq = first_seq[b] + s / 32;
                            const uint32_t sr = s % 32;
                            const uint32_t run = split::min_u32(split::min_u32(valid - r, 16 - r % 16), 16 - sr % 16);
                            const uint32_t run_bytes = run * face_row_bytes;
                            for (uint32_t h = 0; h < 2; ++h) {
                                noc.async_read<NocOptions::TXN_ID>(
                                    window,
                                    ring,
                                    run_bytes,
                                    {.offset_bytes = slot_offset(src_seq) + tile_row_offset(sr, h, face_bytes, face_row_bytes)},
                                    {.offset_bytes = tile_row_offset(r, h, face_bytes, face_row_bytes)},
                                    user_txn);
                            }
                            r += run;
                        }
                    }
                    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                }
#endif
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
