// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_concat writer (BRISC, NoC1): store_block — drains cb_out_tiles block by block into this core's
// contiguous range of output pages. Wait/pop are NOMINAL (block_pages); a ragged last block only
// narrows the writes. One writes_flushed per block, one write barrier at the end (BB-03).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t block_pages = get_compile_time_arg_val(0);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(2);
constexpr auto output_args = TensorAccessorArgs<3>();

void kernel_main() {
    const uint32_t core_start_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    const auto out_acc = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0), tile_bytes);

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    uint32_t remaining = core_pages;
    uint32_t page = core_start_page;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t extent = remaining < block_pages ? remaining : block_pages;
        remaining -= extent;
        {
            DeviceZoneScopedN("writer_wait");
            cb_wait_front(cb_out_tiles, block_pages);
        }
        {
            DeviceZoneScopedN("store_block");
            uint32_t l1_addr = get_read_ptr(cb_out_tiles);
            for (uint32_t i = 0; i < extent; ++i) {
                noc_async_write_page(page++, out_acc, l1_addr);
                l1_addr += tile_bytes;
            }
        }
        {
            DeviceZoneScopedN("writer_flush");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out_tiles, block_pages);
    }
    {
        DeviceZoneScopedN("writer_final_barrier");
        noc_async_write_barrier();
    }
}
