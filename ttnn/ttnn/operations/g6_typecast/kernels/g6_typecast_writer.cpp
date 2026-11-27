// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast writer (BRISC / NoC1): store_block.
//
// Drains cb_output_tiles in blocks of `block_tiles` pages to the same page ids of the interleaved
// output: one wait, `block_extent` async writes, one flush (the CB slot is free once the writes have
// left L1) and one pop per block. Raw dataflow API: no kernel_lib helper writes interleaved tile pages.
// The final write barrier keeps no transaction in flight past kernel exit (BB-03).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);
    const uint32_t page_stride = get_arg_val<uint32_t>(2);

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);

    uint32_t page = first_page;
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t block_extent = (core_tiles - done) < block_tiles ? (core_tiles - done) : block_tiles;
        {
            DeviceZoneScopedN("TC-WR-WAIT");
            cb_wait_front(cb_output_tiles, block_extent);
        }
        {
            DeviceZoneScopedN("TC-WR-ISSUE");
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            for (uint32_t j = 0; j < block_extent; ++j) {
                noc_async_write(l1_read_addr, output.get_noc_addr(page), out_page_bytes);
                l1_read_addr += out_page_bytes;
                page += page_stride;
            }
        }
        {
            DeviceZoneScopedN("TC-WR-BARRIER");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_tiles, block_extent);
    }
    noc_async_write_barrier();
}
