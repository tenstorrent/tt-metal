// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_out0 = 16;

    const uint32_t total_num_rows = get_compile_time_arg_val(2);
    const uint32_t third_dim = get_compile_time_arg_val(3);
    const uint32_t tile_height = get_compile_time_arg_val(4);

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);

    constexpr bool dst0_is_dram = get_compile_time_arg_val(0) == 1;

#if (STICK_SIZE_IS_POW2 == 1)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(1);
    const InterleavedPow2AddrGen<dst0_is_dram> s = {
        .bank_base_address = dst_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<dst0_is_dram> s = {.bank_base_address = dst_addr, .page_size = unpadded_X_size};
#endif
    auto write_block = [&](uint32_t num_rows,
                           uint32_t start_row_id,
                           uint32_t start_column_id,
                           uint32_t width_size,
                           uint32_t size_2d,
                           uint32_t single_block_size) {
        bool has_rows = (num_rows) > 0;

        cb_wait_front(cb_id_out0, single_block_size * has_rows);
        uint32_t l1_read_addr = get_write_ptr(cb_id_out0);

        for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            uint64_t dst_noc_addr = get_noc_addr(size_2d + k, s);
            uint32_t total_size = start_column_id + width_size;
            uint32_t write_size = width_size;

            if (total_size > unpadded_X_size) {
                uint32_t padded_size = total_size - unpadded_X_size;
                write_size -= padded_size;
            }

            noc_async_write(l1_read_addr, dst_noc_addr + start_column_id, write_size);

            noc_async_write_barrier();

            l1_read_addr += width_size;
        }

        cb_pop_front(cb_id_out0, single_block_size * has_rows);
    };

    const uint32_t width_size = get_arg_val<uint32_t>(2);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_row_id = get_arg_val<uint32_t>(3);
        uint32_t start_column_id = get_arg_val<uint32_t>(4);
        uint32_t single_block_size_row_arg = get_arg_val<uint32_t>(5);
        uint32_t single_block_size_col_arg = get_arg_val<uint32_t>(6);

        for (uint32_t b = 0; b < single_block_size_col_arg; b++) {
            uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            if (this_block_num_rows > 0) {
                write_block(
                    this_block_num_rows, start_row_id, start_column_id, width_size, size_2d, single_block_size_row_arg);
            }
            start_row_id += tile_height;
        }
        size_2d += total_num_rows;
    }
}
