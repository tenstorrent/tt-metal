// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    std::uint32_t mem_buffer_dst_addr = get_arg_val<uint32_t>(0);

    std::uint32_t num_rows = get_arg_val<uint32_t>(1);
    std::uint32_t dram_page_size = get_arg_val<uint32_t>(2);

    std::uint32_t vert_expand_count = get_arg_val<uint32_t>(3);
    std::uint32_t skipped_pages = get_arg_val<uint32_t>(4);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t io_cb_id = get_compile_time_arg_val(1);

    InterleavedAddrGen<dst_is_dram> dst_generator = {
        .bank_base_address = mem_buffer_dst_addr,
        .page_size = dram_page_size,
    };

    for (uint32_t i = 0; i < num_rows; i++) {
        cb_wait_front(io_cb_id, 1);
        auto l1_addr = get_read_ptr(io_cb_id);
        for (uint32_t j = 0; j < vert_expand_count; j++) {
            auto noc_addr = get_noc_addr(skipped_pages + j * num_rows + i, dst_generator);
            noc_async_write(l1_addr, noc_addr, dram_page_size);
            noc_async_write_barrier();
        }
        cb_pop_front(io_cb_id, 1);
    }
}
