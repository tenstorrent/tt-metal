// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast writer (BRISC, NoC1) — store_block.
//
// Drains cb_output_tiles into this core's page range of the interleaved output, block_tiles pages
// per CB handshake. Writes of one block are flushed (source L1 read out) before the pop, so the
// slots can be refilled; one final write barrier makes every write land before the kernel returns
// (BB-03: the next program starts right after this one).
//
// CT: [cb_output_tiles, block_tiles, page_bytes, TensorAccessorArgs(output)...]
// common RT: [output_addr]  per-core RT: [start_page, page_stride, core_pages]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t page_stride = get_arg_val<uint32_t>(1);
    const uint32_t core_pages = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, output_addr, page_bytes);

    uint32_t page = start_page;
    for (uint32_t done = 0; done < core_pages; done += block_tiles) {
        DeviceZoneScopedN("store_block");
        const uint32_t remaining = core_pages - done;
        const uint32_t block_extent = remaining < block_tiles ? remaining : block_tiles;

        cb_wait_front(cb_output_tiles, block_extent);
        uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
        for (uint32_t i = 0; i < block_extent; ++i) {
            noc_async_write_page(page, output_accessor, l1_read_addr);
            l1_read_addr += page_bytes;
            page += page_stride;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_tiles, block_extent);
    }
    noc_async_write_barrier();
}
