// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_copy writer (BRISC, NoC1): store_block — cb_page_staging -> destination DRAM pages,
// then drain_writes once.
//
// Raw dataflow, no kernel-lib helper: see g6_copy_reader.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_copy_page_walk.hpp"

template <uint32_t cb_page_staging, uint32_t page_bytes, typename Accessor>
FORCE_INLINE void store_block(
    const Accessor& dst, uint32_t block_first, uint32_t block_len, uint32_t block_pages, uint32_t walk_rotation) {
    {
        DeviceZoneScopedN("G6C-WR-WAIT");
        cb_wait_front(cb_page_staging, block_pages);
    }
    {
        DeviceZoneScopedN("G6C-WR-ISSUE");
        g6_copy::transfer_block<g6_copy::Direction::Write>(
            dst, block_first, block_len, walk_rotation, get_read_ptr(cb_page_staging), page_bytes);
    }
    {
        // The reader may refill this slot as soon as it is popped.
        DeviceZoneScopedN("G6C-WR-FLUSH");
        noc_async_writes_flushed();
    }
    {
        DeviceZoneScopedN("G6C-WR-POP");
        cb_pop_front(cb_page_staging, block_pages);
    }
}

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t page_bytes = dst_args.get_aligned_page_size();

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_pages = get_common_arg_val<uint32_t>(1);

    const uint32_t core_first_page = get_arg_val<uint32_t>(0);
    const uint32_t core_pages = get_arg_val<uint32_t>(1);
    const uint32_t walk_rotation = get_arg_val<uint32_t>(2);

    const auto dst = TensorAccessor(dst_args, dst_addr);

    const uint32_t num_blocks = (core_pages + block_pages - 1) / block_pages;
    const uint32_t last_block_pages = core_pages - (num_blocks - 1) * block_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_len = block_idx + 1 == num_blocks ? last_block_pages : block_pages;
        store_block<cb_page_staging, page_bytes>(
            dst, core_first_page + block_idx * block_pages, block_len, block_pages, walk_rotation);
    }
    {
        // drain_writes: the next program starts right after this one; no write may be in flight.
        DeviceZoneScopedN("G6C-WR-DRAIN");
        noc_async_write_barrier();
    }
}
