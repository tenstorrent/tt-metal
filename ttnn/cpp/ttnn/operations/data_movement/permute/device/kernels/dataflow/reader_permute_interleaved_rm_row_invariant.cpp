// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

struct PermuteReaderStruct {
    bool src_is_dram;
    uint32_t input_rm_page_size;

    constexpr PermuteReaderStruct(bool src_is_dram, uint32_t input_rm_page_size) :
        src_is_dram(src_is_dram), input_rm_page_size(input_rm_page_size) {}
};

void kernel_main() {
    constexpr auto PermuteReaderArgs = PERMUTE_READER_STRUCT;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<PermuteReaderArgs.src_is_dram> s0 = {
        .bank_base_address = src_addr, .page_size = PermuteReaderArgs.input_rm_page_size};

    uint32_t curr_addr = src_addr;
    for (uint32_t row = start_row; row < end_row; ++row) {
        cb_reserve_back(tt::CBIndex::c_0, 1);
        uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
        noc_async_read_page(row, s0, src_buffer_l1_addr);
        noc_async_read_barrier();
        cb_push_back(tt::CBIndex::c_0, 1);
    }
}
