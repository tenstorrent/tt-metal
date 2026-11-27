// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer (BRISC / NoC1): store_tiles_block.
//
// Drains one tile row (block_width_tiles tiles of one unit) per cb_out_tiles handshake and writes the
// tiles as whole result pages: tile id tile_row * tensor_d_tiles + piece * block_width_tiles + c.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_height_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_d_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr bool stagger_writes = get_compile_time_arg_val(6) != 0;
    constexpr auto output_args = TensorAccessorArgs<7>();

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t core_units = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t core_linear_idx = get_arg_val<uint32_t>(3);

    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0));

    // Cores writing the same tile columns at the same time start on different columns, so their
    // first writes go to different DRAM banks.
    const uint32_t start_column = stagger_writes ? core_linear_idx % block_width_tiles : 0;

    uint32_t tile_row = unit_start % tensor_tile_rows;
    uint32_t piece = unit_start / tensor_tile_rows;
    uint32_t units_left = core_units;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t n_units = units_left < block_height_tiles ? units_left : block_height_tiles;
        units_left -= n_units;
        // Tile rows past n_units are the phantom rows of a ragged last block: popped, never written.
        for (uint32_t unit = 0; unit < block_height_tiles; ++unit) {
            {
                DeviceZoneScopedN("TILE-WAIT");
                cb_wait_front(cb_out_tiles, block_width_tiles);
            }
            if (unit < n_units) {
                const uint32_t l1_base = get_read_ptr(cb_out_tiles);
                const uint32_t first_tile = tile_row * tensor_d_tiles + piece * block_width_tiles;
                {
                    DeviceZoneScopedN("TILE-WRITE");
                    uint32_t column = start_column;
                    for (uint32_t c = 0; c < block_width_tiles; ++c) {
                        noc_async_write(
                            l1_base + column * tile_bytes, output.get_noc_addr(first_tile + column), tile_bytes);
                        if (++column == block_width_tiles) {
                            column = 0;
                        }
                    }
                }
                {
                    // The slot is reusable once the data has left L1; acks are collected once at the end.
                    DeviceZoneScopedN("WRITE-FLUSH");
                    noc_async_writes_flushed();
                }
                if (++tile_row == tensor_tile_rows) {
                    tile_row = 0;
                    ++piece;
                }
            }
            cb_pop_front(cb_out_tiles, block_width_tiles);
        }
    }

    DeviceZoneScopedN("WRITE-BARRIER");
    noc_async_write_barrier();
}
