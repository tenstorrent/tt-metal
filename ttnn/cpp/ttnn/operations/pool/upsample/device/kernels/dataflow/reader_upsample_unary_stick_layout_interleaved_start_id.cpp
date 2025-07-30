// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr bool src_page_size_is_pow2 = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(4);

    const auto s0 =
        get_interleaved_addr_gen<src0_is_dram, src_page_size_is_pow2>(src_addr, page_size, src_log_base_2_of_page_size);

    const uint32_t end_id = start_page_id + num_pages;

    // reader copied the data from DRAM to CB buffer.
    for (uint32_t i = start_page_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        uint64_t src_noc_addr = get_noc_addr(i, s0);

        noc_async_read(src_noc_addr, l1_write_addr, page_size);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, 1);
    }
}
