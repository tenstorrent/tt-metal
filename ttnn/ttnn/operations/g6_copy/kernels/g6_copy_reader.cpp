// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_copy reader (NCRISC, NoC0): load_block — source DRAM pages -> cb_page_staging.
//
// Raw dataflow, no kernel-lib helper: no helper moves a multi-burst DRAM span into a CB
// (local_copy_helpers are L1->L1 and capped at one burst), and a copy has no compute phase.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_copy_page_walk.hpp"

template <uint32_t cb_page_staging, uint32_t page_bytes, typename Accessor>
FORCE_INLINE void load_block(
    const Accessor& src, uint32_t block_first, uint32_t block_len, uint32_t block_pages, uint32_t walk_rotation) {
    {
        DeviceZoneScopedN("G6C-RD-RESERVE");
        cb_reserve_back(cb_page_staging, block_pages);
    }
    {
        DeviceZoneScopedN("G6C-RD-ISSUE");
        g6_copy::transfer_block<g6_copy::Direction::Read>(
            src, block_first, block_len, walk_rotation, get_write_ptr(cb_page_staging), page_bytes);
    }
    {
        DeviceZoneScopedN("G6C-RD-BARRIER");
        noc_async_read_barrier();
    }
    {
        // Nominal count even on a short last block: the CB ring closes only on whole blocks.
        DeviceZoneScopedN("G6C-RD-PUSH");
        cb_push_back(cb_page_staging, block_pages);
    }
}

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    constexpr uint32_t page_bytes = src_args.get_aligned_page_size();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(1);

    const uint32_t core_first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    const uint32_t walk_rotation = get_arg_val<uint32_t>(2);

    const auto src = TensorAccessor(src_args, src_addr);

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    const uint32_t last_block_pages = core_pages - (num_blocks - 1) * block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_len = block_idx + 1 == num_blocks ? last_block_pages : block_pages;
        load_block<cb_page_staging, page_bytes>(
            src, core_first_page + block_idx * block_pages, block_len, block_pages, walk_rotation);
    }
}
