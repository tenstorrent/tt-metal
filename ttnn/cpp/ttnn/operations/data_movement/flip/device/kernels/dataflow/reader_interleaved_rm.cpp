// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "hello from reader kernel" << ENDL();
    // Compile time args
    constexpr bool src_is_dram = static_cast<bool>(get_compile_time_arg_val(0));
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t row_width = get_compile_time_arg_val(2);
    constexpr uint32_t rank = get_compile_time_arg_val(3);

    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    uint32_t dims_to_flip[rank];
    for (uint32_t i = 0; i < rank; i++) {
        dims_to_flip[i] = get_arg_val<uint32_t>(i + 4);
    }

    // Derived constants
    const bool is_vertical_flip = static_cast<bool>(dims_to_flip[rank - 2]);
    const bool is_horizontal_flip = static_cast<bool>(dims_to_flip[rank - 1]);

    const InterleavedAddrGen<src_is_dram> s0 = {.bank_base_address = src_addr, .page_size = page_size};

    // if (is_vertical_flip == true) read rows in reverse order
    // else read rows in normal order
    int32_t step = is_vertical_flip ? -1 : 1;
    int32_t start = is_vertical_flip ? end_row - 1 : start_row;
    int32_t end = is_vertical_flip ? -1 : end_row;
    DPRINT << start << "-" << end << ENDL();
    for (int32_t row_id = start; row_id != end; row_id += step) {
        DPRINT << row_id << ENDL();
        cb_reserve_back(tt::CBIndex::c_0, 1);
        uint32_t l1_buffer_addr = get_write_ptr(tt::CBIndex::c_0);
        uint64_t read_noc_addr = get_noc_addr(row_id, s0);
        noc_async_read(read_noc_addr, l1_buffer_addr, page_size);
        noc_async_read_barrier();

        for (uint32_t col_id = 0; col_id < row_width; ++col_id) {
            DPRINT << uint32_t(reinterpret_cast<uint32_t*>(l1_buffer_addr)[col_id]) << ", ";
        }
        DPRINT << ENDL();

        // if (is_horizontal_flip) {
        //     // flip elements within the row
        //     uint32_t* row_data = reinterpret_cast<uint32_t*>(l1_buffer_addr);
        //     for (uint32_t col_id = 0; i < row_width / 2; ++col_id) {
        //         uint32_t temp = row_data[col_id];
        //         row_data[col_id] = row_data[row_width - 1 - col_id];
        //         row_data[row_width - 1 - col_id] = temp;
        //     }
        // }
        cb_push_back(tt::CBIndex::c_0, 1);
    }
}
