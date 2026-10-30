// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm writer: drains one tile-row (block_w_tiles tiles) per handshake; ragged tile-rows
// (>= core_rows) are popped without being written.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t output_tile_bytes = get_compile_time_arg_val(3);
    constexpr auto output_args = TensorAccessorArgs<4>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_rows = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks = get_arg_val<uint32_t>(3);

    const auto output_accessor = TensorAccessor(output_args, output_addr, output_tile_bytes);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t row = 0; row < block_rows; ++row) {
            const uint32_t core_row = block * block_rows + row;
            cb_wait_front(cb_output_tiles, block_w_tiles);
            if (core_row < core_rows) {
                uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
                const uint32_t first_tile = (start_row_tile + core_row) * block_w_tiles;
                for (uint32_t col = 0; col < block_w_tiles; ++col) {
                    noc_async_write(l1_read_addr, output_accessor.get_noc_addr(first_tile + col), output_tile_bytes);
                    l1_read_addr += output_tile_bytes;
                }
                noc_async_writes_flushed();
            }
            cb_pop_front(cb_output_tiles, block_w_tiles);
        }
    }
    noc_async_write_barrier();
}
