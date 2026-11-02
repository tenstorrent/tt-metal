// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// linear_gelu writer (BRISC, NoC1): store_output_block. Drains cb_output_tiles per subblock in the
// matmul's subblock-major order (subblock row i, subblock column j, then tiles row-major inside it),
// writing out page m * n_tiles + n. Flush before each pop, one barrier at the end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_h = get_compile_time_arg_val(1);
    constexpr uint32_t block_w = get_compile_time_arg_val(2);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_rows = get_arg_val<uint32_t>(2);
    const uint32_t n_tile_start = get_arg_val<uint32_t>(3);
    const uint32_t num_n_blocks = get_arg_val<uint32_t>(4);
    const uint32_t sb_h = get_arg_val<uint32_t>(5);
    const uint32_t sb_w = get_arg_val<uint32_t>(6);
    const uint32_t tail_sb_h = get_arg_val<uint32_t>(7);
    const uint32_t tail_sb_w = get_arg_val<uint32_t>(8);

    const auto out_accessor = TensorAccessor(out_args, out_addr, tile_bytes);

    const uint32_t num_row_blocks = (core_rows + block_h - 1) / block_h;
    for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
        const uint32_t first_row = row_block_idx * block_h;
        const bool tail = core_rows - first_row < block_h;
        const uint32_t rows = tail ? core_rows - first_row : block_h;
        const uint32_t cur_sb_h = tail ? tail_sb_h : sb_h;
        const uint32_t cur_sb_w = tail ? tail_sb_w : sb_w;
        const uint32_t sb_tiles = cur_sb_h * cur_sb_w;
        const uint32_t m_base = row_start + first_row;

        for (uint32_t n_block_idx = 0; n_block_idx < num_n_blocks; ++n_block_idx) {
            const uint32_t n_base = n_tile_start + n_block_idx * block_w;
            for (uint32_t sb_row = 0; sb_row < rows; sb_row += cur_sb_h) {
                for (uint32_t sb_col = 0; sb_col < block_w; sb_col += cur_sb_w) {
                    cb_wait_front(cb_output_tiles, sb_tiles);
                    uint32_t l1_addr = get_read_ptr(cb_output_tiles);
                    for (uint32_t r = 0; r < cur_sb_h; ++r) {
                        const uint32_t page_row = (m_base + sb_row + r) * n_tiles + n_base + sb_col;
                        for (uint32_t c = 0; c < cur_sb_w; ++c) {
                            noc_async_write_page(page_row + c, out_accessor, l1_addr);
                            l1_addr += tile_bytes;
                        }
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_output_tiles, sb_tiles);
                }
            }
        }
    }
    noc_async_write_barrier();
}
