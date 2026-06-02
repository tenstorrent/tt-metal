// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the full-row e4m3 -> fp32 tilize pipeline.
// The compute kernel produces, per (tile-row, column-block), a 32-page batch of fp32 row-major
// output (one page per row, 4096 bytes = 1024 fp32 elements). We consume each batch in the same
// order and write each page to its row's DRAM page at the column-block byte offset.
//   for tr: for c: cb_wait_front(32); for s in 0..31 -> write row (tr*32+s) at offset c*4096.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    uint32_t num_col_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_tile_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out_fp32 = get_compile_time_arg_val(0);
    constexpr uint32_t col_block_bytes = get_compile_time_arg_val(1);  // 4096 (fp32)
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        for (uint32_t c = 0; c < num_col_blocks; ++c) {
            cb_wait_front(cb_out_fp32, TILE_HEIGHT);
            uint32_t l1 = get_read_ptr(cb_out_fp32);
            uint32_t col_offset_bytes = c * col_block_bytes;
            for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
                uint32_t page_id = (start_tile_row + tr) * TILE_HEIGHT + s;
                noc_async_write(l1, dst.get_noc_addr(page_id) + col_offset_bytes, col_block_bytes);
                l1 += col_block_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out_fp32, TILE_HEIGHT);
        }
    }
}
