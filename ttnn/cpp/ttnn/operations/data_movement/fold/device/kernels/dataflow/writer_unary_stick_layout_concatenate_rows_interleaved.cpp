// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_page_size = get_arg_val<uint32_t>(1);

    const uint32_t scratch_addr = get_arg_val<uint32_t>(2);

    const uint32_t pixel_size = get_arg_val<uint32_t>(3);
    const uint32_t aligned_pixel_size = get_arg_val<uint32_t>(4);
    const uint32_t aligned_chunk_size = get_arg_val<uint32_t>(5);
    const uint32_t aligned_row_size = get_arg_val<uint32_t>(6);

    const uint32_t stride_h = get_arg_val<uint32_t>(7);
    const uint32_t stride_w = get_arg_val<uint32_t>(8);

    const uint32_t num_dst_rows = get_arg_val<uint32_t>(9);
    const uint32_t num_dst_cols = get_arg_val<uint32_t>(10);
    const uint32_t cb_pages_per_dst_row = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr bool stick_size_is_power_of_two = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(3);

    const auto s = get_interleaved_addr_gen<dst_is_dram, stick_size_is_power_of_two>(
        dst_addr, dst_page_size, log_base_2_of_page_size);

    auto dst_noc_addr = NOC_XY_ADDR(NOC_X(my_x[0]), NOC_Y(my_y[0]), scratch_addr);

    auto extract_next_dst_page = [&](uint32_t src_address, uint64_t dst_noc_addr) -> uint32_t {
        for (uint32_t row = 0, src_row_offset = 0; row < stride_h; ++row, src_row_offset += aligned_row_size) {
            for (uint32_t col = 0, src_col_offset = 0; col < stride_w; ++col, src_col_offset += aligned_pixel_size) {
                uint32_t src_offset = src_col_offset + src_row_offset;
                noc_async_write(src_address + src_offset, dst_noc_addr, pixel_size);
                dst_noc_addr += pixel_size;
            }
        }

        return src_address + aligned_chunk_size;
    };

    uint32_t dst_page_id = 0;
    for (uint32_t i = 0; i < num_dst_rows; ++i) {
        cb_wait_front(cb_id_out0, cb_pages_per_dst_row);
        uint32_t src_addr = get_read_ptr(cb_id_out0);

        for (uint32_t j = 0; j < num_dst_cols; ++j) {
            src_addr = extract_next_dst_page(src_addr, dst_noc_addr);
            uint64_t dst_addr = get_noc_addr(dst_page_id, s);
            noc_async_write(scratch_addr, dst_addr, dst_page_size);
            dst_page_id += 1;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, cb_pages_per_dst_row);
    }
}
