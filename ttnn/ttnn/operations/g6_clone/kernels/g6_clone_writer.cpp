// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_clone writer (BRISC, NoC1): store_block over this core's contiguous page range,
// then drain_writes. Raw dataflow for the same reason as the reader.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr uint32_t block_pages = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    const auto dst = TensorAccessor(dst_args, dst_addr);
    const uint32_t page_bytes = dst.get_aligned_page_size();

    for (uint32_t block_offset = 0; block_offset < num_pages; block_offset += block_pages) {
        const uint32_t remaining = num_pages - block_offset;
        const uint32_t block_extent = remaining < block_pages ? remaining : block_pages;
        const uint32_t first_page = start_page + block_offset;

        {
            DeviceZoneScopedN("g6c_wr_wait");
            cb_wait_front(cb_page_staging, block_pages);
        }
        uint32_t l1_read_addr = get_read_ptr(cb_page_staging);
        {
            DeviceZoneScopedN("g6c_wr_issue");
            for (uint32_t i = 0; i < block_extent; ++i) {
                noc_async_write_page(first_page + i, dst, l1_read_addr);
                l1_read_addr += page_bytes;
            }
        }
        // The slot is handed back to the reader only after its bytes have left L1.
        {
            DeviceZoneScopedN("g6c_wr_flush");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_page_staging, block_pages);
    }

    // Every write must be acknowledged before the kernel returns: the next program starts right after.
    {
        DeviceZoneScopedN("g6c_wr_drain");
        noc_async_write_barrier();
    }
}
