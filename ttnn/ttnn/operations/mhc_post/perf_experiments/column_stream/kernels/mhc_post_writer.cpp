// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post writer (BRISC, NoC1) — COLUMN-STREAM candidate (perf_experiments/column_stream).
//
// store_group: per group of G columns (host CT arg), wait n*G output tiles (slot j*G + c), write the valid ones
// to X'_j page r*n*Ct + j*Ct + col, noc_async_writes_flushed (the data has left L1: the slots may be reused)
// and pop the nominal n*G. One write barrier at kernel end (acks), instead of one per block.
// load_coefficients (when the host's COEF_EXPANDER knob names the writer, expand_here): the writer is idle
// until compute's first output block, so it owns the coefficient read + expansion (mhc_post_coef_expand.hpp)
// and keeps it off the reader's streaming path. Segment 0's set is loaded up front; segment s+1's is loaded
// right after segment s's first block is written (a look-ahead walker over the same SegmentWalker
// derivation), so it is ready before compute reaches s+1. Needs COEF_DEPTH >= 2 (host asserts).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "mhc_post_common.hpp"
#include "mhc_post_coef_expand.hpp"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);  // = G (group_col_tiles)
    constexpr uint32_t output_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t post_tiles_per_row = get_compile_time_arg_val(5);
    constexpr uint32_t comb_tiles_per_row = get_compile_time_arg_val(6);
    constexpr uint32_t coef_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t tile_rows = get_compile_time_arg_val(8);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(9);
    constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(10);
    constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(11);
    constexpr bool expand_here = get_compile_time_arg_val(12) != 0;     // COEF_EXPANDER == writer
    constexpr bool flush_only = get_compile_time_arg_val(13) != 0;      // per-group flush (1) or barrier (0) before pop
    constexpr uint32_t write_col_tiles = get_compile_time_arg_val(14);  // WG: columns per writer window (multiple of G)
    constexpr uint32_t tail_cols = get_compile_time_arg_val(15);        // trailing columns written one group at a time
    constexpr auto output_args = TensorAccessorArgs<16>();
    constexpr auto post_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    static_assert(tile_rows == 2 * mhc_post::FACE_HW, "mhc_post: expansion assumes 32x32 tiles of 16x16 faces");

    constexpr uint32_t output_row_tiles = n * col_tiles_per_row;
    constexpr uint32_t output_block_tiles = n * block_col_tiles;  // one compute group (n*G tiles, slot j*G + c)
    static_assert(write_col_tiles % block_col_tiles == 0, "writer window = whole compute groups");

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);
    const uint32_t post_addr = get_arg_val<uint32_t>(3);
    const uint32_t comb_addr = get_arg_val<uint32_t>(4);

    const auto output_acc = TensorAccessor(output_args, output_addr, output_page_bytes);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_page_bytes);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page_bytes);
    mhc_post::CoefExpander<
        n,
        coef_tiles_per_stream,
        post_tiles_per_row,
        comb_tiles_per_row,
        coef_page_bytes,
        cb_coef_raw,
        cb_coef_bcast,
        decltype(post_acc)>
        coefs(post_acc, comb_acc);

    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);
    mhc_post::SegmentWalker ahead = walker;  // coefficient look-ahead (same derivation)
    if constexpr (expand_here) {
        if (!ahead.done()) {
            MaybeDeviceZoneScope("writer_coef_load");
            coefs.load(ahead.next().row);
        }
    }
    // Writer window: WG columns = WG/G compute groups (a ragged segment tail has fewer groups). Tile (j, c) of the
    // window sits in group c/G at slot j*G + c%G; the window may straddle the CB wrap (ragged windows shift the
    // alignment), so every tile address is wrapped individually.
    const uint32_t fifo_limit = get_local_cb_interface(cb_output_tiles).fifo_limit;
    const uint32_t fifo_size = get_local_cb_interface(cb_output_tiles).fifo_size;
    uint32_t units_left = num_units;
    while (!walker.done()) {
        const mhc_post::Segment seg = walker.next();
        for (uint32_t done = 0; done < seg.col_tiles;) {
            // Window: WG columns, or one group within the core's last tail_cols columns (fine-grained tail).
            const uint32_t width = units_left <= tail_cols ? block_col_tiles : write_col_tiles;
            const uint32_t valid = seg.col_tiles - done < width ? seg.col_tiles - done : width;
            const uint32_t col_start = seg.col0 + done;
            const bool first_window = done == 0;
            done += valid;
            units_left -= valid;
            const uint32_t tiles = mhc_post::num_blocks(valid, block_col_tiles) * output_block_tiles;

            {
                MaybeDeviceZoneScope("writer_wait");  // starved on compute
                cb_wait_front(cb_output_tiles, tiles);
            }
            {
                MaybeDeviceZoneScope("writer_issue");  // issue + flush
                const uint32_t out_base = get_read_ptr(cb_output_tiles);
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t page0 = seg.row * output_row_tiles + j * col_tiles_per_row + col_start;
                    for (uint32_t c = 0; c < valid; ++c) {
                        const uint32_t slot =
                            (c / block_col_tiles) * output_block_tiles + j * block_col_tiles + c % block_col_tiles;
                        uint32_t addr = out_base + slot * output_page_bytes;
                        if (addr >= fifo_limit) {
                            addr -= fifo_size;
                        }
#ifndef CS_STUB_DM
                        noc_async_write(addr, output_acc.get_noc_addr(page0 + c), output_page_bytes);
#endif
                    }
                }
                if constexpr (flush_only) {
                    noc_async_writes_flushed();
                } else {
                    noc_async_write_barrier();
                }
            }
            for (uint32_t t = 0; t < tiles; t += output_block_tiles) {
                cb_pop_front(
                    cb_output_tiles, output_block_tiles);  // group-aligned pops: each lands on the wrap exactly
            }

            if constexpr (expand_here) {
                if (first_window && !ahead.done()) {
                    MaybeDeviceZoneScope("writer_coef_load");
                    coefs.load(ahead.next().row);  // next segment's set, ahead of compute
                }
            }
        }
    }
    {
        MaybeDeviceZoneScope("writer_barrier");
        noc_async_write_barrier();
    }
}
