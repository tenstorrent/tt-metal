// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t output_unit_size = get_arg_val<uint32_t>(3);

    constexpr uint32_t dst_cb_id = tt::CBIndex::c_16;
    constexpr uint32_t src_cb_id = tt::CBIndex::c_0;
    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t onetile = 1;

    const InterleavedAddrGen<output_is_dram> s = {
        .bank_base_address = output_buffer_address,
        .page_size = output_unit_size,
    };
    for (uint32_t i = start_id; i < start_id + num_rows_per_core; i++) {
        cb_wait_front(src_cb_id, onetile);

        uint32_t writer_ptr = get_read_ptr(src_cb_id);
        uint64_t output_noc_addr = get_noc_addr(i, s);
        noc_async_write(writer_ptr, output_noc_addr, output_unit_size);
        noc_async_write_barrier();

        cb_pop_front(src_cb_id, onetile);
    }
}
