// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_clone reader (NCRISC, NoC0): load_block over this core's contiguous page range.
//
// Raw dataflow by design: no kernel_lib helper moves DRAM pages into a CB behind one
// barrier per block (op_design.md, "Helpers considered and rejected"). The bytes never
// reach an unpacker, so every bit pattern of every dtype survives.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

// One block: reserve the nominal block, issue every page read back to back, one barrier,
// push the nominal block. A ragged block narrows only the reads, so every push is
// block_pages and the CB ring closes on its capacity (cb_depth_blocks * block_pages).
template <uint32_t cb_page_staging, uint32_t block_pages, typename Accessor>
FORCE_INLINE void load_block(const Accessor& x_accessor, uint32_t first_page, uint32_t block_extent) {
    {
        DeviceZoneScopedN("g6c_rd_reserve");
        cb_reserve_back(cb_page_staging, block_pages);
    }
    {
        DeviceZoneScopedN("g6c_rd_issue");
        const uint32_t page_bytes = x_accessor.get_aligned_page_size();
        uint32_t l1_write_addr = get_write_ptr(cb_page_staging);
        for (uint32_t i = 0; i < block_extent; ++i) {
            noc_async_read_page(first_page + i, x_accessor, l1_write_addr);
            l1_write_addr += page_bytes;
        }
    }
    {
        DeviceZoneScopedN("g6c_rd_barrier");
        noc_async_read_barrier();
    }
    cb_push_back(cb_page_staging, block_pages);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr uint32_t block_pages = get_compile_time_arg_val(1);
    constexpr auto x_args = TensorAccessorArgs<2>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages_this_core = get_arg_val<uint32_t>(1);

    // Page size comes from the accessor args (the buffer's aligned page), the same value
    // the host used as the CB page size.
    const auto x_accessor = TensorAccessor(x_args, x_addr);

    const uint32_t num_blocks_this_core = (num_pages_this_core + block_pages - 1) / block_pages;
    uint32_t first_page = start_page;
    uint32_t pages_left = num_pages_this_core;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t block_extent = pages_left < block_pages ? pages_left : block_pages;
        load_block<cb_page_staging, block_pages>(x_accessor, first_page, block_extent);
        first_page += block_extent;
        pages_left -= block_extent;
    }
}
