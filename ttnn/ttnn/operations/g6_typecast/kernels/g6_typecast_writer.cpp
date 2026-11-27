// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast writer (BRISC / NoC1) — block operation `store_block` (op_design.md, Block schedule).
// Drains cb_output_tiles in blocks of `block_tiles` (a ragged last block waits/pops its actual
// extent, matching reader and compute), writes the tiles of each block to [start_tile, start_tile + core_tiles),
// flushes before releasing the slot, and ends with one write barrier so no transaction is in flight at exit (BB-03).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);

    const auto output_accessor = TensorAccessor(output_args, output_addr, out_page_bytes);

    const uint32_t end_tile = start_tile + core_tiles;
    for (uint32_t block_start = start_tile; block_start < end_tile; block_start += block_tiles) {
        DeviceZoneScopedN("WRITE_BLOCK");
        const uint32_t block_extent = (end_tile - block_start < block_tiles) ? (end_tile - block_start) : block_tiles;
        {
            DeviceZoneScopedN("WRITE_WAIT_DATA");
            cb_wait_front(cb_output_tiles, block_extent);
        }
        uint32_t l1_addr = get_read_ptr(cb_output_tiles);
        for (uint32_t t = 0; t < block_extent; ++t) {
            noc_async_write_page(block_start + t, output_accessor, l1_addr);
            l1_addr += out_page_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_tiles, block_extent);
    }
    noc_async_write_barrier();
}
