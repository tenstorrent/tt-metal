// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t page_size = get_compile_time_arg_val(src_args.next_compile_time_args_offset());

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_page_id = get_arg_val<uint32_t>(2);

    const auto s = TensorAccessor(src_args, src_addr, page_size);

    // Read pages (rows) from source and write to circular buffer
    for (uint32_t i = 0; i < num_pages; ++i) {
        uint32_t page_id = start_page_id + i;
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        uint64_t src_noc_addr = get_noc_addr(page_id, s);
        noc_async_read(src_noc_addr, l1_write_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
    }
}
