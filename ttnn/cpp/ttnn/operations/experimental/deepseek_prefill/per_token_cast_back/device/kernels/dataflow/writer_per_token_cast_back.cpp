// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the full-row e4m3 -> fp32 tilize pipeline.
// The compute kernel produces, per (tile-row, column-block), a 32-page batch of fp32 row-major
// output (one page per row, 4096 bytes = 1024 fp32 elements). We consume each batch in the same
// order and write each page to its row's DRAM page at the column-block byte offset.
//   for tr: for c: cb_wait_front(32); for s in 0..31 -> write row (tr*32+s) at offset c*4096.

#include <cstdint>
#include <algorithm>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    uint32_t num_col_blocks = get_arg_val<uint32_t>(2);
    uint32_t start_tile_row = get_arg_val<uint32_t>(3);
    uint32_t m_total = get_arg_val<uint32_t>(4);  // total rows (M); last tile-row may be partial
    uint32_t h_total = get_arg_val<uint32_t>(5);  // total width (H); last col-block may be partial

    constexpr uint32_t cb_out_fp32 = get_compile_time_arg_val(0);
    constexpr uint32_t col_block_bytes = get_compile_time_arg_val(1);  // COL_BLOCK_ELEMS * out_elem_size
    // Tile height from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t COL_BLOCK_ELEMS = 1024;                              // LLK column-block width
    constexpr uint32_t out_elem_bytes = col_block_bytes / COL_BLOCK_ELEMS;  // output element size
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        const uint32_t row_base = (start_tile_row + tr) * tile_h;
        uint32_t rows_this = std::min(tile_h, m_total - row_base);  // real rows in this tile-row
        for (uint32_t c = 0; c < num_col_blocks; ++c) {
            uint32_t real_col_elems = std::min(COL_BLOCK_ELEMS, h_total - c * COL_BLOCK_ELEMS);
            uint32_t real_col_bytes = real_col_elems * out_elem_bytes;  // real output width
            // Drain the full CB, but only write real rows / real columns.
            cb_wait_front(cb_out_fp32, tile_h);
            uint32_t l1 = get_read_ptr(cb_out_fp32);
            uint32_t col_offset_bytes = c * col_block_bytes;
            for (uint32_t s = 0; s < tile_h; ++s) {
                if (s < rows_this) {
                    noc_async_write(l1, dst.get_noc_addr(row_base + s) + col_offset_bytes, real_col_bytes);
                }
                l1 += col_block_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out_fp32, tile_h);
        }
    }
}
