// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config — STAGE regime writer. BRISC / NoC1 on every core of the full-grid split.
// Drains `cb_staged_pages` block by block: wait one nominal block, issue one NoC write per maximal
// run, flush (departure — the slot may then be reused), pop the nominal block. One write barrier
// at kernel end guarantees arrival before the program ends.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_to_memory_config_common.hpp"

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t stage_block_pages = get_compile_time_arg_val(1);
    constexpr uint32_t cb_staged_pages = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    uint32_t arg = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t start_page = get_arg_val<uint32_t>(arg++);
    const uint32_t core_pages = get_arg_val<uint32_t>(arg++);

    const auto dst = TensorAccessor(dst_args, dst_addr, page_size);
    const uint32_t num_blocks = (core_pages + stage_block_pages - 1) / stage_block_pages;

    uint32_t page = start_page;
    uint32_t remaining = core_pages;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_pages = remaining < stage_block_pages ? remaining : stage_block_pages;
        {
            DeviceZoneScopedN("wait_staged_block");
            cb_wait_front(cb_staged_pages, stage_block_pages);
        }
        {
            DeviceZoneScopedN("store_block");
            g6_tmc::write_span(dst, page, block_pages, get_read_ptr(cb_staged_pages), page_size);
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_staged_pages, stage_block_pages);
        page += block_pages;
        remaining -= block_pages;
    }
    noc_async_write_barrier();
}
