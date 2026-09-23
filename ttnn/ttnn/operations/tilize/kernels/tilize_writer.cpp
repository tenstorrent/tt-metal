// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize writer (BRISC / NoC1) — the `store_block` block operation (op_design.md).
//
// No kernel_lib dataflow helper writes TILE pages (write_sticks_after_untilize
// writes ROW_MAJOR sticks), so this is a custom block operation.
//
// Per tile-row of a column block: wait block_width pages -> valid_width tile-page
// writes (all in flight) -> one flush (L1 source reads done) -> pop block_width.
// The flush, rather than a full write barrier, is enough to release the CB
// slots; one write barrier at kernel end guarantees the data has landed.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);     // tiles per column block (CB quantum)
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(2);  // one output TILE page
    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(4);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(5);  // C: output tile-columns of the whole tensor

    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);

    const uint32_t num_col_blocks = (core_col_tiles + block_width - 1) / block_width;
    const uint32_t row_end = row_start + core_row_tiles;

    for (uint32_t col_block_idx = 0; col_block_idx < num_col_blocks; ++col_block_idx) {
        const uint32_t block_col = col_block_idx * block_width;  // relative to col_start
        const uint32_t remaining = core_col_tiles - block_col;
        const uint32_t valid_width = remaining < block_width ? remaining : block_width;

        for (uint32_t row = row_start; row < row_end; ++row) {
            cb_wait_front(cb_output_tiles, block_width);
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            uint32_t tile_idx = row * tiles_per_row + col_start + block_col;
            for (uint32_t t = 0; t < valid_width; ++t) {
                noc_async_write(l1_read_addr, output_accessor.get_noc_addr(tile_idx), out_tile_bytes);
                l1_read_addr += out_tile_bytes;
                ++tile_idx;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_output_tiles, block_width);
        }
    }
    noc_async_write_barrier();
}
