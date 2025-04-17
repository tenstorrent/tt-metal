// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_out0 = 16;

    const uint32_t total_num_rows = get_compile_time_arg_val(3);
    const uint32_t ncores = get_compile_time_arg_val(4);
    const uint32_t third_dim = get_compile_time_arg_val(5);
    const uint32_t tile_width = get_compile_time_arg_val(6);

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t core_number = get_arg_val<uint32_t>(2);

    constexpr bool dst0_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr bool stick_size_is_pow2 = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);

    const auto s =
        get_interleaved_addr_gen<dst0_is_dram, stick_size_is_pow2>(dst_addr, unpadded_X_size, log_base_2_of_page_size);

    auto write_block = [&](uint32_t num_rows,
                           uint32_t mul,
                           uint32_t size_per_row_per_block,
                           uint32_t start_id,
                           uint32_t width_size,
                           uint32_t size_2d) {
        uint32_t onetile = 1;
        bool has_rows = (num_rows) > 0;

        cb_wait_front(cb_id_out0, onetile * has_rows);
        uint32_t l1_read_addr = get_write_ptr(cb_id_out0);

        for (uint32_t k = 0; k < num_rows; k++) {
            uint64_t dst_noc_addr = get_noc_addr(size_2d + k, s);

            uint32_t total_size = mul * size_per_row_per_block + start_id + width_size;
            uint32_t padded_size = total_size - unpadded_X_size;
            uint32_t write_size = width_size;

            if (mul == ncores - 1 && padded_size > 0) {
                write_size = width_size - padded_size;
            }

            noc_async_write(l1_read_addr, dst_noc_addr + start_id + mul * size_per_row_per_block, write_size);

            noc_async_write_barrier();

            if (k > 0 && (k % tile_width == 0)) {
                cb_pop_front(cb_id_out0, onetile * has_rows);
                cb_wait_front(cb_id_out0, onetile * has_rows);
            }
            l1_read_addr += width_size;
        }

        cb_pop_front(cb_id_out0, onetile * has_rows);
    };

    const uint32_t size_per_row_per_block = get_arg_val<uint32_t>(3);
    const uint32_t blocks_per_core = get_arg_val<uint32_t>(4);
    const uint32_t width_size = get_arg_val<uint32_t>(5);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_id = 0;
        for (uint32_t b = 0; b < blocks_per_core; b++) {
            write_block(total_num_rows, core_number, size_per_row_per_block, start_id, width_size, size_2d);
            start_id += width_size;
        }
        size_2d += total_num_rows;
    }
}
