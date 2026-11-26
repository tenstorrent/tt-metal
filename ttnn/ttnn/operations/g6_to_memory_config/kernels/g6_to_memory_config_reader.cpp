// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config — STAGE regime reader (both ends are banks: interleaved DRAM/L1 or
// DRAM-sharded). NCRISC / NoC0 on every core of the full-grid split. Loads this core's contiguous
// page range block by block into the staging ring `cb_staged_pages`: reserve one nominal block,
// issue one NoC read per maximal run, one read barrier, push the nominal block (a ragged last block
// transfers fewer pages but still pushes nominal, keeping ring pointers block-aligned).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_to_memory_config_common.hpp"

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t stage_block_pages = get_compile_time_arg_val(1);
    constexpr uint32_t cb_staged_pages = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    uint32_t arg = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start_page = get_arg_val<uint32_t>(arg++);
    const uint32_t core_pages = get_arg_val<uint32_t>(arg++);

    const auto src = TensorAccessor(src_args, src_addr, page_size);
    const uint32_t num_blocks = (core_pages + stage_block_pages - 1) / stage_block_pages;

    uint32_t page = start_page;
    uint32_t remaining = core_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_pages = remaining < stage_block_pages ? remaining : stage_block_pages;
        {
            DeviceZoneScopedN("wait_free_slot");
            cb_reserve_back(cb_staged_pages, stage_block_pages);
        }
        {
            DeviceZoneScopedN("load_block");
            g6_tmc::read_span(src, page, block_pages, get_write_ptr(cb_staged_pages), page_size);
            noc_async_read_barrier();
        }
        cb_push_back(cb_staged_pages, stage_block_pages);
        page += block_pages;
        remaining -= block_pages;
    }
}
