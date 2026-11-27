// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_clone reader (NCRISC, NoC0): load_block over this core's contiguous page range.
//
// Raw dataflow, no kernel_lib helper: no helper moves a DRAM page range into a CB as
// raw bytes behind one barrier (the compute copy helpers route through DEST, which
// changes float32 bits; the stick helpers hand off one page per row).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_page_staging = get_compile_time_arg_val(0);
    constexpr uint32_t block_pages = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);

    const auto src = TensorAccessor(src_args, src_addr);
    const uint32_t page_bytes = src.get_aligned_page_size();

    for (uint32_t block_offset = 0; block_offset < num_pages; block_offset += block_pages) {
        const uint32_t remaining = num_pages - block_offset;
        const uint32_t block_extent = remaining < block_pages ? remaining : block_pages;
        const uint32_t first_page = start_page + block_offset;

        // Reserve and push the nominal block_pages even on a ragged block so the ring
        // pointers land exactly on fifo_limit; only the transfers narrow.
        {
            DeviceZoneScopedN("g6c_rd_reserve");
            cb_reserve_back(cb_page_staging, block_pages);
        }
        uint32_t l1_write_addr = get_write_ptr(cb_page_staging);
        {
            DeviceZoneScopedN("g6c_rd_issue");
            for (uint32_t i = 0; i < block_extent; ++i) {
                noc_async_read_page(first_page + i, src, l1_write_addr);
                l1_write_addr += page_bytes;
            }
        }
        {
            DeviceZoneScopedN("g6c_rd_barrier");
            noc_async_read_barrier();
        }
        cb_push_back(cb_page_staging, block_pages);
    }
}
