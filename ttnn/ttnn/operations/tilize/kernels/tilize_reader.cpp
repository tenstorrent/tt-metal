// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize reader (NCRISC / NoC0) — the `load_block` block operation (op_design.md).
//
// Custom block operation, not dataflow_kernel_lib::read_sticks_for_tilize: that
// helper derives BOTH the CB push quantum and the L1 stick stride from the valid
// row bytes, so narrowing a ragged last column block would push fewer than
// block_width pages and break the CB ring-wrap invariant
// (tilize_helpers_dataflow.inl:92-93, 117, 127; see op_design.md API Mapping).
//
// Per Tensix core: output-tile rectangle [row_start, row_start + core_row_tiles)
// x [col_start, col_start + core_col_tiles), cut into column blocks of
// block_width tiles. Per tile-row of a block:
//   reserve block_width pages -> tile_h stick-segment reads (all in flight)
//   -> ONE read barrier -> push block_width pages.
// The push/pop quantum is always the nominal block_width; only the NoC transfer
// narrows to valid_width on the ragged last column block.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);       // tiles per column block (CB quantum)
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);            // sticks per tile-row
    constexpr uint32_t tile_col_bytes = get_compile_time_arg_val(3);    // bytes of one stick per tile-column
    constexpr uint32_t stick_page_bytes = get_compile_time_arg_val(4);  // aligned interleaved stick page
    constexpr auto input_args = TensorAccessorArgs<5>();

    // L1 stride between consecutive sticks inside one tile-row slot (nominal width).
    constexpr uint32_t block_stick_bytes = block_width * tile_col_bytes;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(4);

    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);

    const uint32_t num_col_blocks = (core_col_tiles + block_width - 1) / block_width;
    const uint32_t row_end = row_start + core_row_tiles;

    for (uint32_t col_block_idx = 0; col_block_idx < num_col_blocks; ++col_block_idx) {
        const uint32_t block_col = col_block_idx * block_width;  // relative to col_start
        const uint32_t remaining = core_col_tiles - block_col;
        const uint32_t valid_width = remaining < block_width ? remaining : block_width;
        const uint32_t segment_bytes = valid_width * tile_col_bytes;
        const uint32_t segment_offset = (col_start + block_col) * tile_col_bytes;

        for (uint32_t row = row_start; row < row_end; ++row) {
            cb_reserve_back(cb_input_sticks, block_width);
            uint32_t l1_write_addr = get_write_ptr(cb_input_sticks);
            uint32_t stick_idx = row * tile_h;
            for (uint32_t s = 0; s < tile_h; ++s) {
                noc_async_read(input_accessor.get_noc_addr(stick_idx, segment_offset), l1_write_addr, segment_bytes);
                l1_write_addr += block_stick_bytes;
                ++stick_idx;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_sticks, block_width);
        }
    }
}
