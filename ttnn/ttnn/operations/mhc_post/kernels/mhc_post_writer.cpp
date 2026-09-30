// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post writer (BRISC, NoC1).
//
// store_block: per block, wait n*B output tiles (slot j*B + c), write the valid ones to
// X'_j page r*n*Ct + j*Ct + col, ONE write barrier, pop the nominal n*B.
// load_coefficients (when the host's COEF_EXPANDER knob names the writer, expand_here): the writer is idle
// until compute's first output block, so it owns the coefficient read + expansion (mhc_post_coef_expand.hpp)
// and keeps it off the reader's streaming path. Segment 0's set is loaded up front; segment s+1's is loaded
// right after segment s's first block is written (a look-ahead walker over the same SegmentWalker
// derivation), so it is ready before compute reaches s+1. Needs COEF_DEPTH >= 2 (host asserts).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_post_common.hpp"
#include "mhc_post_coef_expand.hpp"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t post_tiles_per_row = get_compile_time_arg_val(5);
    constexpr uint32_t comb_tiles_per_row = get_compile_time_arg_val(6);
    constexpr uint32_t coef_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t tile_rows = get_compile_time_arg_val(8);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(9);
    constexpr uint32_t cb_coef_bcast = get_compile_time_arg_val(10);
    constexpr uint32_t coef_tiles_per_stream = get_compile_time_arg_val(11);
    constexpr bool expand_here = get_compile_time_arg_val(12) != 0;  // COEF_EXPANDER == writer
    constexpr auto output_args = TensorAccessorArgs<13>();
    constexpr auto post_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    static_assert(tile_rows == 2 * mhc_post::FACE_HW, "mhc_post: expansion assumes 32x32 tiles of 16x16 faces");

    constexpr uint32_t output_row_tiles = n * col_tiles_per_row;
    constexpr uint32_t output_block_tiles = n * block_col_tiles;

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
            coefs.load(ahead.next().row);
        }
    }
    while (!walker.done()) {
        const mhc_post::Segment seg = walker.next();
        const uint32_t blocks = mhc_post::num_blocks(seg.col_tiles, block_col_tiles);
        for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
            const uint32_t valid = mhc_post::block_valid_col_tiles(seg.col_tiles, block_col_tiles, block_idx);
            const uint32_t col_start = seg.col0 + block_idx * block_col_tiles;

            cb_wait_front(cb_output_tiles, output_block_tiles);
            const uint32_t out_base = get_read_ptr(cb_output_tiles);
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t page0 = seg.row * output_row_tiles + j * col_tiles_per_row + col_start;
                const uint32_t slot0 = out_base + j * block_col_tiles * output_page_bytes;
#ifdef ABL_NO_DM
                if (true) {
                    continue;
                }
#endif
                for (uint32_t c = 0; c < valid; ++c) {
                    noc_async_write(
                        slot0 + c * output_page_bytes, output_acc.get_noc_addr(page0 + c), output_page_bytes);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, output_block_tiles);

            if constexpr (expand_here) {
                if (block_idx == 0 && !ahead.done()) {
                    coefs.load(ahead.next().row);  // next segment's set, ahead of compute
                }
            }
        }
    }
}
