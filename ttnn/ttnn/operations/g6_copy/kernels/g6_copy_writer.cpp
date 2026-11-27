// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// store_block: cb_page_staging -> destination DRAM, one block of block_pages pages at a time,
// then drain_writes once. Raw dataflow by design: no kernel-lib helper writes multi-burst DRAM bank
// runs from a CB.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_copy_page_walk.hpp"

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(1);

    const uint32_t core_first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    const uint32_t walk_rotation = get_arg_val<uint32_t>(2);

    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    uint32_t block_first = core_first_page;
    uint32_t pages_left = core_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_len = pages_left < block_pages ? pages_left : block_pages;
        {
            DeviceZoneScopedN("G6C-WR-WAIT");
            cb_wait_front(cb_page_staging, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-WR-ISSUE");
            g6_copy::transfer_block<false>(dst, block_first, block_len, walk_rotation, get_read_ptr(cb_page_staging));
        }
        // The slot may be refilled by the reader once popped, so its bytes must have left L1 first.
        {
            DeviceZoneScopedN("G6C-WR-FLUSH");
            noc_async_writes_flushed();
        }
        {
            DeviceZoneScopedN("G6C-WR-POP");
            cb_pop_front(cb_page_staging, block_pages);
        }
        block_first += block_len;
        pages_left -= block_len;
    }

    // The next program starts right after this one; no write may still be in flight.
    {
        DeviceZoneScopedN("G6C-WR-DRAIN");
        noc_async_write_barrier();
    }
}
