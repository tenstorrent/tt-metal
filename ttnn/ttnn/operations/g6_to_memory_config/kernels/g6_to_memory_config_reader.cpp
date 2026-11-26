// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// staged reader (DRAM -> DRAM only, NCRISC/NoC0): a NoC transfer cannot go DRAM -> DRAM, so the pages
// pass through cb_staged_pages. Loads this core's contiguous page range in blocks of
// staged_block_pages; the writer drains block k while block k + 1 fills.
//
// Raw NoC calls, no kernel_lib helper: see g6_to_memory_config_move.hpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/ttnn/operations/g6_to_memory_config/kernels/g6_to_memory_config_move.hpp"

void kernel_main() {
    constexpr uint32_t cb_staged_pages = get_compile_time_arg_val(0);
    constexpr uint32_t staged_block_pages = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    const uint32_t src_base = get_common_arg_val<uint32_t>(0);
    const auto src = TensorAccessor(src_args, src_base);
    const uint32_t page_size = src.get_aligned_page_size();

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = (num_pages + staged_block_pages - 1) / staged_block_pages;

    uint32_t page = start_page;
    uint32_t remaining = num_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_pages = remaining < staged_block_pages ? remaining : staged_block_pages;
        // CB counts stay nominal on the ragged last block so the ring cycle closes; only the NoC
        // transfer carries the actual count.
        {
            DeviceZoneScopedN("G6-STAGED-RESERVE");
            cb_reserve_back(cb_staged_pages, staged_block_pages);
        }
        {
            DeviceZoneScopedN("G6-STAGED-READ");
            g6_to_memory_config::move_page_range<true>(
                src, page, block_pages, get_write_ptr(cb_staged_pages), page_size);
        }
        {
            DeviceZoneScopedN("G6-STAGED-READ-BARRIER");
            noc_async_read_barrier();
        }
        cb_push_back(cb_staged_pages, staged_block_pages);
        page += block_pages;
        remaining -= block_pages;
    }
}
