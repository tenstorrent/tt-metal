// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_2;
    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr uint32_t page_size = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_page_id = get_arg_val<uint32_t>(2);

    const auto s = TensorAccessor(dst_args, dst_addr, page_size);

    // Read pages from circular buffer and write to destination
    for (uint32_t i = 0; i < num_pages; ++i) {
        uint32_t page_id = start_page_id + i;
        cb_wait_front(cb_id_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        uint64_t dst_noc_addr = get_noc_addr(page_id, s);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
