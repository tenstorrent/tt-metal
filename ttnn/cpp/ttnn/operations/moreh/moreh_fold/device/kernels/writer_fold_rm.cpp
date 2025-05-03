// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_cb_page_size = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t num_units_per_core = get_arg_val<uint32_t>(3);
    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(1);

    constexpr int onetile = 1;

    const InterleavedAddrGen<output_is_dram> s = {.bank_base_address = output_addr, .page_size = output_cb_page_size};

    for (uint32_t i = start_id; i < start_id + num_units_per_core; i++) {
        cb_wait_front(output_cb_id, onetile);
        uint32_t l1_read_addr = get_read_ptr(output_cb_id);
        uint64_t dst_noc_addr = get_noc_addr(i, s);
        noc_async_write(l1_read_addr, dst_noc_addr, output_cb_page_size);
        noc_async_write_barrier();
        cb_pop_front(output_cb_id, onetile);
    }
}
