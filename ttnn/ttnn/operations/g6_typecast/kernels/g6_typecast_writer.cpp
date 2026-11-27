// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast writer (BRISC / NoC1) — store_block.
//
// Mirrors the reader: block_extent page writes issued back to back per block, to the same page
// ids the reader loaded. A block only needs its writes to have left L1 before the slot is popped
// (async_writes_flushed); completion is awaited once, at kernel end, so the next program never
// starts with a write of this one in flight.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_tiles = get_common_arg_val<uint32_t>(1);
    const uint32_t page_stride = get_common_arg_val<uint32_t>(2);

    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t last_block_extent = get_arg_val<uint32_t>(2);

    const auto output = TensorAccessor(output_args, output_addr, out_page_bytes);
    Noc noc;
    DataflowBuffer output_tiles(cb_output_tiles);

    uint32_t page = first_page;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_extent = (block_idx + 1 == num_blocks) ? last_block_extent : block_tiles;
        {
            DeviceZoneScopedN("store_block.wait");
            output_tiles.wait_front(block_tiles);
        }
        {
            DeviceZoneScopedN("store_block.write");
            for (uint32_t i = 0; i < block_extent; ++i) {
                noc.async_write(
                    output_tiles, output, out_page_bytes, {.offset_bytes = i * out_page_bytes}, {.page_id = page});
                page += page_stride;
            }
            noc.async_writes_flushed();
        }
        output_tiles.pop_front(block_tiles);
    }
    {
        DeviceZoneScopedN("store.write_barrier");
        noc.async_write_barrier();
    }
}
