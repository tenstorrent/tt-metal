// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast reader (NCRISC, NoC0) — load_block.
//
// Streams this core's page range [start_page, start_page + core_pages·page_stride) of the
// interleaved input into cb_input_tiles, block_tiles pages per CB handshake and per NoC barrier.
// Every push is block_tiles pages except the core's last block (the final push of the kernel),
// so the CB ring never sees a mid-stream partial push.
//
// CT: [cb_input_tiles, block_tiles, page_bytes, TensorAccessorArgs(input)...]
// common RT: [input_addr]   per-core RT: [start_page, page_stride, core_pages]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t page_stride = get_arg_val<uint32_t>(1);
    const uint32_t core_pages = get_arg_val<uint32_t>(2);

    const auto input_accessor = TensorAccessor(input_args, input_addr, page_bytes);

    uint32_t page = start_page;
    for (uint32_t done = 0; done < core_pages; done += block_tiles) {
        DeviceZoneScopedN("load_block");
        const uint32_t remaining = core_pages - done;
        const uint32_t block_extent = remaining < block_tiles ? remaining : block_tiles;

        cb_reserve_back(cb_input_tiles, block_extent);
        uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
        for (uint32_t i = 0; i < block_extent; ++i) {
            noc_async_read_page(page, input_accessor, l1_write_addr);
            l1_write_addr += page_bytes;
            page += page_stride;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, block_extent);
    }
}
