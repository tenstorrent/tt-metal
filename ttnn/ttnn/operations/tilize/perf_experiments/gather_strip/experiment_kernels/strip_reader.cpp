// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gather_strip bench reader — the cross-core L1 gather, WITH the real pipeline
// behind it (compute + writer are present in this bench, unlike Perf 1's
// reader-only isolation).
//
// Geometry: a ROW_MAJOR source whose shard is NARROWER than a tensor row, so a
// row is `row_pages` pages living on `row_pages` DIFFERENT cores' L1. This core
// assembles blocks of `tile_h` rows x `row_bytes` bytes into cb_input_sticks.
//
// MODES (compile-time arg 0):
//
//   0 ROW    — the op's CURRENT approach (tilize_reader.cpp R_PAD, read_row_span):
//              one transfer per (source row, page slice). tile_h * slices
//              transfers per block, destination laid out ROW-MAJOR.
//
//   1 STRIP  — one transfer per (block, source shard). Consecutive rows of one
//              source shard are consecutive pages of that shard, hence CONTIGUOUS
//              in the source core's L1; laying them down STRIP-MAJOR (all tile_h
//              rows of shard s, then of shard s+1) makes the destination
//              contiguous too, so the whole strip is ONE transfer of
//              tile_h * page_bytes. `slices` transfers per block.
//
//              COMPUTE-SIDE CONTRACT (this is what Perf 1 never exercised):
//              strip s of a block is itself a row-major region of tile_h rows x
//              `page_tiles`*32 elements, i.e. exactly a `page_tiles`-wide tilize
//              block, and it holds tile-columns [s*page_tiles, (s+1)*page_tiles)
//              of the block. So the library helper consumes the strip layout
//              unchanged, simply called at block_width_tiles = page_tiles over
//              slices x more blocks — the emitted TILE SEQUENCE is identical.
//              Nothing about the helper had to change.
//
//   2 STRIP_FINE — same transfers as 1, but the CB handshake is per STRIP
//              (reserve/barrier/push `page_tiles` pages) instead of per block, so
//              compute can start on strip s while strip s+1 is still in flight.
//
// RAW DATAFLOW, deliberately (same justification as the production kernel):
// `dataflow_kernel_lib::read_sticks_for_tilize` walks CONSECUTIVE PAGE IDS as
// consecutive sticks, which is exactly the identity a narrower-than-a-row shard
// breaks (page p -> shard p % row_pages, local row p / row_pages). The gather is
// not expressible through it in EITHER mode, which is why the production op
// hand-rolls it too.

//
// PAD COMPOSITION (compile-time `h_in` < the padded row count): the op's R_PAD
// reader assembles the fill INTO the CB as it reads. The strip form composes with
// it without touching the fill at all, because raggedness is per BLOCK:
//   * a block whose tile_h rows are ALL real (row0 + tile_h <= h_in) is a plain
//     strip — the fast form, and that is every block but the last tile-row.
//   * the ragged block falls back to per-(strip,row) transfers + fill, laid down
//     at the SAME STRIP-MAJOR addresses. That is the load-bearing detail: the
//     fallback must not revert to a row-major slot, or the strip-width tilize
//     behind it would read the block transposed. Same layout, slower issue.
// So the fast form still covers every whole block of a padded call; only the H
// tail (at most one tile-row per core) pays the row-form price.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
// The op's own alignment-aware L1 fill — reused verbatim so the pad arm's fill
// cost is the production one, not a bench approximation.
#include "ttnn/ttnn/operations/tilize/kernels/tilize_fill.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = 0;

    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t tile_h = get_compile_time_arg_val(1);
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t row_pages = get_compile_time_arg_val(5);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t page_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t slices = get_compile_time_arg_val(8);
    // Rows of REAL data in the source (== the padded row count off the pad path,
    // which compiles every fill branch below away).
    constexpr uint32_t h_in = get_compile_time_arg_val(9);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(10);
    constexpr auto src_args = TensorAccessorArgs<11>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row0 = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t pad_word = get_arg_val<uint32_t>(3);

    // A strip is only ONE transfer when the block width is a whole number of
    // source shards. The host only selects modes 1/2 where that holds.
    static_assert(mode == 0 || row_bytes == slices * page_bytes, "strip needs whole-page slices");

    if (num_blocks == 0) {
        return;
    }

    const auto accessor = TensorAccessor(src_args, src_addr);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t r_blk = b / n_chunks;
        const uint32_t c_blk = b - r_blk * n_chunks;
        const uint32_t row0 = (tile_row0 + r_blk) * tile_h;
        const uint32_t col_off = c_blk * row_bytes;
        const uint32_t page_in_row0 = col_off / page_bytes;

        // Is every row of this block real? (Always true off the pad path.)
        const bool whole_block = (row0 + tile_h <= h_in);

        if constexpr (mode == 2) {
            // ── STRIP, per-strip handshake ───────────────────────────────
            for (uint32_t s = 0; s < slices; ++s) {
                {
                    MaybeDeviceZoneScope("reader_reserve");
                    cb_reserve_back(cb_input_sticks, page_tiles);
                }
                {
                    MaybeDeviceZoneScope("reader_issue");
                    const uint32_t a0 = get_write_ptr(cb_input_sticks);
                    if (whole_block) {
                        noc_async_read(
                            accessor.get_noc_addr(row0 * row_pages + page_in_row0 + s, 0), a0, tile_h * page_bytes);
                    } else {
                        for (uint32_t r = 0; r < tile_h; ++r) {
                            const uint32_t a = a0 + r * page_bytes;
                            if (row0 + r < h_in) {
                                noc_async_read(
                                    accessor.get_noc_addr((row0 + r) * row_pages + page_in_row0 + s, 0), a, page_bytes);
                            } else {
                                tilize_kernels::fill_l1_with_val<elem_bytes>(a, page_bytes, pad_word);
                            }
                        }
                    }
                }
                {
                    MaybeDeviceZoneScope("reader_barrier");
                    noc_async_read_barrier();
                }
                cb_push_back(cb_input_sticks, page_tiles);
            }
            continue;
        }

        {
            MaybeDeviceZoneScope("reader_reserve");
            cb_reserve_back(cb_input_sticks, wt_chunk);
        }
        const uint32_t dst = get_write_ptr(cb_input_sticks);

        {
            MaybeDeviceZoneScope("reader_issue");
            if constexpr (mode == 0) {
                // ── ROW: the production R_PAD / read_row_span loop ────────
                uint32_t l1_addr = dst;
                for (uint32_t r = 0; r < tile_h; ++r) {
                    const uint32_t row = row0 + r;
                    if (row < h_in) {
                        uint32_t page = row * row_pages + page_in_row0;
                        uint32_t off = col_off - page_in_row0 * page_bytes;
                        uint32_t n_bytes = row_bytes;
                        uint32_t a = l1_addr;
                        while (n_bytes > 0) {
                            uint32_t n = page_bytes - off;
                            if (n > n_bytes) {
                                n = n_bytes;
                            }
                            noc_async_read(accessor.get_noc_addr(page, off), a, n);
                            a += n;
                            n_bytes -= n;
                            ++page;
                            off = 0;
                        }
                    } else {
                        tilize_kernels::fill_l1_with_val<elem_bytes>(l1_addr, row_bytes, pad_word);
                    }
                    l1_addr += row_bytes;
                }
            } else if (whole_block) {
                // ── STRIP: one transfer per (block, source shard) ─────────
                for (uint32_t s = 0; s < slices; ++s) {
                    noc_async_read(
                        accessor.get_noc_addr(row0 * row_pages + page_in_row0 + s, 0),
                        dst + s * tile_h * page_bytes,
                        tile_h * page_bytes);
                }
            } else {
                // ── STRIP, RAGGED tail block: per-(strip,row) transfers +
                //    fill, into the SAME STRIP-MAJOR addresses.
                for (uint32_t s = 0; s < slices; ++s) {
                    uint32_t a = dst + s * tile_h * page_bytes;
                    for (uint32_t r = 0; r < tile_h; ++r) {
                        if (row0 + r < h_in) {
                            noc_async_read(
                                accessor.get_noc_addr((row0 + r) * row_pages + page_in_row0 + s, 0), a, page_bytes);
                        } else {
                            tilize_kernels::fill_l1_with_val<elem_bytes>(a, page_bytes, pad_word);
                        }
                        a += page_bytes;
                    }
                }
            }
        }
        {
            MaybeDeviceZoneScope("reader_barrier");
            noc_async_read_barrier();  // ONE barrier per block (the op's policy)
        }
        cb_push_back(cb_input_sticks, wt_chunk);
    }
}
