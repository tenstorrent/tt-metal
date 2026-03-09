// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Compile-time arguments
constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

void kernel_main() {
    // Runtime arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);
    uint32_t buffer_page_size = get_arg_val<uint32_t>(2);
    uint32_t num_buffer_pages = get_arg_val<uint32_t>(3);

    // Set up address generator with actual buffer page size
    const InterleavedAddrGenFast<dst_is_dram> dst_addr_gen = {
        .bank_base_address = dst_addr,
        .page_size = buffer_page_size,
        .data_format = DataFormat::Float16_b
    };

    // Wait for all tiles to be computed
    cb_wait_front(cb_out0, n_tiles);

    uint32_t cb_out0_base_addr = get_read_ptr(cb_out0);

    // Write all pages using proper interleaved addressing
    uint32_t l1_read_offset = 0;
    for (uint32_t page_id = 0; page_id < num_buffer_pages; page_id++) {
        uint64_t dst_noc_addr = get_noc_addr(page_id, dst_addr_gen);
        noc_async_write(cb_out0_base_addr + l1_read_offset, dst_noc_addr, buffer_page_size);
        l1_read_offset += buffer_page_size;
    }
    noc_async_write_barrier();

    // Pop all tiles
    cb_pop_front(cb_out0, n_tiles);
}
