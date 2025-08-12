// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, uint32_t val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {
    constexpr uint32_t cb_id_in0 = 0;

    const uint32_t total_num_rows = get_compile_time_arg_val(2);
    const uint32_t third_dim = get_compile_time_arg_val(3);
    const uint32_t tile_height = get_compile_time_arg_val(4);
    const uint32_t element_size = get_compile_time_arg_val(5);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t pad_value = get_arg_val<uint32_t>(2);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

#if (STICK_SIZE_IS_POW2 == 1)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(1);
    const InterleavedPow2AddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<src0_is_dram> s = {.bank_base_address = src_addr, .page_size = unpadded_X_size};
#endif

    auto read_block = [&](uint32_t num_rows,
                          uint32_t start_row_id,
                          uint32_t start_column_id,
                          uint32_t width_size,
                          uint32_t size_2d,
                          uint32_t element_size,
                          uint32_t single_block_size) {
        uint32_t padding_rows = num_rows == 32 ? 0 : 32 - num_rows;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_reserve_back(cb_id_in0, single_block_size * has_rows);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        uint32_t original_addr = get_write_ptr(cb_id_in0);
        for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            uint64_t src_noc_addr = get_noc_addr(size_2d + k, s);

            // Read from DRAM to tmp buffer
            noc_async_read(src_noc_addr + start_column_id, l1_write_addr, width_size);

            uint32_t prev_size = start_column_id;
            uint32_t this_block_size = unpadded_X_size - prev_size;
            if (this_block_size < width_size) {
                uint32_t to_pad = width_size - this_block_size;
                fill_with_val(l1_write_addr + this_block_size + element_size, (to_pad) >> 2, pad_value);
            }

            // Block before copying data from tmp to cb buffer
            noc_async_read_barrier();
            l1_write_addr += width_size;
        }

        for (uint32_t pad_row = 0; pad_row < padding_rows; pad_row++) {
            fill_with_val(l1_write_addr, (width_size >> 2), pad_value);
            l1_write_addr += width_size;
        }

        cb_push_back(cb_id_in0, single_block_size * has_rows);
    };

    const uint32_t width_size = get_arg_val<uint32_t>(3);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_row_id = get_arg_val<uint32_t>(4);
        uint32_t start_column_id = get_arg_val<uint32_t>(5);
        uint32_t single_block_size_row_arg = get_arg_val<uint32_t>(6);
        uint32_t single_block_size_col_arg = get_arg_val<uint32_t>(7);
        for (uint32_t b = 0; b < single_block_size_col_arg; b++) {
            uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            if (this_block_num_rows > 0) {
                read_block(
                    this_block_num_rows,
                    start_row_id,
                    start_column_id,
                    width_size,
                    size_2d,
                    element_size,
                    single_block_size_row_arg);
            }
            start_row_id += tile_height;
        }
        size_2d += total_num_rows;
    }
}
