// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // compile time args
    constexpr bool dst_is_dram = get_named_compile_time_arg_val("dst_is_dram") == 1;
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr auto dst_args = TensorAccessorArgs<0>();

    // runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_row = get_arg_val<uint32_t>(1);
    uint32_t end_row = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    // const InterleavedAddrGen<dst_is_dram> d0 = {.bank_base_address = dst_addr, .page_size = page_size};
    const auto d0 = TensorAccessor(dst_args, dst_addr, page_size);

    for (uint32_t row_id = start_row; row_id < end_row; ++row_id) {
        cb_wait_front(cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        uint64_t dst_noc_addr = get_noc_addr(row_id, d0);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
