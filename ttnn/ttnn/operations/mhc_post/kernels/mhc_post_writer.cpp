// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post writer (BRISC, NoC1).
//
// store_block: per block, wait n*B output tiles (slot j*B + c), write the valid ones to
// X'_j page r*n*Ct + j*Ct + col, ONE write barrier, pop the nominal n*B.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_post_common.hpp"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();

    constexpr uint32_t output_row_tiles = n * col_tiles_per_row;
    constexpr uint32_t output_block_tiles = n * block_col_tiles;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    const auto output_acc = TensorAccessor(output_args, output_addr, output_page_bytes);

    mhc_post::SegmentWalker walker(start_unit, num_units, col_tiles_per_row);
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
                for (uint32_t c = 0; c < valid; ++c) {
                    noc_async_write(
                        slot0 + c * output_page_bytes, output_acc.get_noc_addr(page0 + c), output_page_bytes);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, output_block_tiles);
        }
    }
}
