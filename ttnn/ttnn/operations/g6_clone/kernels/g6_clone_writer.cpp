// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_clone writer (BRISC, NoC1): store_block over this core's contiguous page range, then
// drain_writes.
//
// Raw dataflow by design: no kernel_lib helper writes a CB block to DRAM pages behind one
// flush per block (op_design.md, "Helpers considered and rejected").

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

// One block: wait for the nominal block, issue every page write back to back, flush so the
// slot's bytes have left L1 before the reader may refill it, pop the nominal block.
template <uint32_t cb_page_staging, uint32_t block_pages, typename Accessor>
FORCE_INLINE void store_block(const Accessor& out_accessor, uint32_t first_page, uint32_t block_extent) {
    {
        DeviceZoneScopedN("g6c_wr_wait");
        cb_wait_front(cb_page_staging, block_pages);
    }
    {
        DeviceZoneScopedN("g6c_wr_issue");
        const uint32_t page_bytes = out_accessor.get_aligned_page_size();
        uint32_t l1_read_addr = get_read_ptr(cb_page_staging);
        for (uint32_t i = 0; i < block_extent; ++i) {
            noc_async_write_page(first_page + i, out_accessor, l1_read_addr);
            l1_read_addr += page_bytes;
        }
    }
    {
        DeviceZoneScopedN("g6c_wr_flush");
        noc_async_writes_flushed();
    }
    cb_pop_front(cb_page_staging, block_pages);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr uint32_t block_pages = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages_this_core = get_arg_val<uint32_t>(1);

    const auto out_accessor = TensorAccessor(out_args, out_addr);

    const uint32_t num_blocks_this_core = (num_pages_this_core + block_pages - 1) / block_pages;
    uint32_t first_page = start_page;
    uint32_t pages_left = num_pages_this_core;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t block_extent = pages_left < block_pages ? pages_left : block_pages;
        store_block<cb_page_staging, block_pages>(out_accessor, first_page, block_extent);
        first_page += block_extent;
        pages_left -= block_extent;
    }

    // The next program starts right after this one returns: every write must be acked.
    {
        DeviceZoneScopedN("g6c_wr_drain");
        noc_async_write_barrier();
    }
}
