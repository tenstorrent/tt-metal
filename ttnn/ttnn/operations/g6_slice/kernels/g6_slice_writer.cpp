// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_slice writer (BRISC, NoC1): store_block.
//
// Helper policy: no kernel_lib helper covers a contiguous page write from a CB to an interleaved
// tensor with block-granular handshakes; raw dataflow API per op_design.md -> API Mapping.
//
// Per block of `block_tiles` (CT knob): 1 cb_wait_front(block_tiles), `extent` page writes to
// contiguous output page ids, 1 noc_async_writes_flushed, 1 cb_pop_front(block_tiles).
// One noc_async_write_barrier at kernel end (BB-03: nothing in flight at return).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);

    const auto output = TensorAccessor(output_args, out_addr, page_size);

    const uint32_t num_blocks = (core_tiles + block_tiles - 1) / block_tiles;
    uint32_t page = start_tile;
    uint32_t remaining = core_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t extent = remaining < block_tiles ? remaining : block_tiles;
        remaining -= extent;
        {
            DeviceZoneScopedN("writer_wait");
            cb_wait_front(cb_output_tiles, block_tiles);
        }
        {
            DeviceZoneScopedN("store_block");
            uint32_t src = get_read_ptr(cb_output_tiles);
            for (uint32_t j = 0; j < extent; ++j) {
                noc_async_write_page(page, output, src);
                src += page_size;
                ++page;
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_tiles, block_tiles);
    }
    {
        DeviceZoneScopedN("write_barrier");
        noc_async_write_barrier();
    }
}
