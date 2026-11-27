// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// load_block: source DRAM -> cb_page_staging, one block of block_pages pages at a time.
// Raw dataflow by design: no kernel-lib helper reads multi-burst DRAM bank runs into a CB.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_copy_page_walk.hpp"

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(1);

    const uint32_t core_first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    const uint32_t walk_rotation = get_arg_val<uint32_t>(2);

    const auto src = TensorAccessor(src_args, src_addr);

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    uint32_t block_first = core_first_page;
    uint32_t pages_left = core_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_len = pages_left < block_pages ? pages_left : block_pages;
        // The nominal block_pages is reserved and pushed even on a short last block, so the ring
        // (capacity cb_depth_blocks * block_pages) never wraps inside a block.
        {
            DeviceZoneScopedN("G6C-RD-RESERVE");
            cb_reserve_back(cb_page_staging, block_pages);
        }
        {
            DeviceZoneScopedN("G6C-RD-ISSUE");
            g6_copy::transfer_block<true>(src, block_first, block_len, walk_rotation, get_write_ptr(cb_page_staging));
        }
        {
            DeviceZoneScopedN("G6C-RD-BARRIER");
            noc_async_read_barrier();
        }
        {
            DeviceZoneScopedN("G6C-RD-PUSH");
            cb_push_back(cb_page_staging, block_pages);
        }
        block_first += block_len;
        pages_left -= block_len;
    }
}
