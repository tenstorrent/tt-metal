// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast writer (BRISC / NoC1) — store_block.
//
// Drains cb_output_tiles to the same page ids of the interleaved output (DRAM or L1),
// block_tiles pages per handshake and one completion wait per block. Raw dataflow API, mirror of
// the reader.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_tiles = get_arg_val<uint32_t>(1);
    const uint32_t page_stride = get_arg_val<uint32_t>(2);

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);

    const uint32_t num_blocks = (core_tiles + block_tiles - 1) / block_tiles;
    uint32_t page = first_page;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_extent =
            (block_idx + 1 == num_blocks) ? core_tiles - block_idx * block_tiles : block_tiles;
        {
            DeviceZoneScopedN("TC-WR-WAIT");
            cb_wait_front(cb_output_tiles, block_extent);
        }
        {
            DeviceZoneScopedN("TC-WR-ISSUE");
            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            for (uint32_t j = 0; j < block_extent; ++j) {
                noc_async_write(l1_read_addr, dst.get_noc_addr(page), page_bytes);
                l1_read_addr += page_bytes;
                page += page_stride;
            }
        }
        {
            // The slot may be refilled once its bytes have left L1; landing is awaited once below.
            DeviceZoneScopedN("TC-WR-BARRIER");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_output_tiles, block_extent);
    }
    // No write may be in flight when the next program starts (BB-03).
    noc_async_write_barrier();
}
