// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer (BRISC, NoC1): store_tiles_block.
//
// Per unit (tile row t, column piece p): col_block_tiles whole-tile writes to output pages
// t * Dt + p * col_block_tiles + j. Writes are only flushed per unit (the source slots may be
// reused once issued); the single write barrier at the end retires every write before return.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t col_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t unit_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    const auto out_accessor = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = get_tile_size(cb_tiles);

    uint32_t piece = unit_begin / tensor_tile_rows;
    uint32_t tile_row = unit_begin - piece * tensor_tile_rows;

    for (uint32_t unit = 0; unit < num_units; ++unit) {
        {
            DeviceZoneScopedN("tiles-wait");
            cb_wait_front(cb_tiles, col_block_tiles);
        }
        {
            DeviceZoneScopedN("tiles-write");
            uint32_t l1_addr = get_read_ptr(cb_tiles);
            const uint32_t first_page = tile_row * tensor_col_tiles + piece * col_block_tiles;
            for (uint32_t j = 0; j < col_block_tiles; ++j) {
                noc_async_write(l1_addr, out_accessor.get_noc_addr(first_page + j), tile_bytes);
                l1_addr += tile_bytes;
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_tiles, col_block_tiles);

        if (++tile_row == tensor_tile_rows) {
            tile_row = 0;
            ++piece;
        }
    }

    {
        DeviceZoneScopedN("write-barrier");
        noc_async_write_barrier();
    }
}
