// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer (RISCV_0, NoC1): store_tiles_block, whole 2 KiB tiles to tile id
// tile_row * width_tiles + width_block * block_width_tiles + w. Raw NoC: no kernel_lib helper
// stores whole TILE pages (write_sticks_after_untilize writes row-major sticks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t width_blocks_per_row = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr auto output_args = TensorAccessorArgs<6>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    const uint32_t end_unit = first_unit + num_units;
    uint32_t tile_row = first_unit / width_blocks_per_row;
    uint32_t width_block = first_unit % width_blocks_per_row;

    for (uint32_t block_first = first_unit; block_first < end_unit; block_first += block_tile_rows) {
        const uint32_t units_left = end_unit - block_first;
        const uint32_t block_units = units_left < block_tile_rows ? units_left : block_tile_rows;
        const uint32_t block_pages = block_units * block_width_tiles;

        {
            DeviceZoneScopedN("STORE_WAIT_IN");
            cb_wait_front(cb_output_tiles, block_pages);
        }
        {
            DeviceZoneScopedN("STORE_ISSUE");
            uint32_t tile_l1 = get_read_ptr(cb_output_tiles);
            for (uint32_t u = 0; u < block_units; ++u) {
                const uint32_t first_tile = tile_row * width_tiles + width_block * block_width_tiles;
                for (uint32_t w = 0; w < block_width_tiles; ++w) {
                    noc_async_write_page(first_tile + w, output_accessor, tile_l1);
                    tile_l1 += tile_bytes;
                }
                if (++width_block == width_blocks_per_row) {
                    width_block = 0;
                    ++tile_row;
                }
            }
        }
        {
            // The slots are reused only after their bytes have left L1.
            DeviceZoneScopedN("STORE_FLUSH");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_tiles, block_pages);
    }

    {
        DeviceZoneScopedN("STORE_BARRIER");
        noc_async_write_barrier();
    }
}
