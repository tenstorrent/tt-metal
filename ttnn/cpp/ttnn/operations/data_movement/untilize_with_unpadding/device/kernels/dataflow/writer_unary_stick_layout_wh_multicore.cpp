// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr uint32_t cb_id_out0 = 16;
    experimental::CircularBuffer cb_out0(cb_id_out0);
    experimental::Noc noc;

    constexpr uint32_t total_num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t third_dim = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t unpadded_X_size = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    const auto s = TensorAccessor(dst_args, dst_addr);
    auto write_block = [&](uint32_t num_rows,
                           uint32_t start_row_id,
                           uint32_t start_column_id,
                           uint32_t width_size,
                           uint32_t size_2d,
                           uint32_t single_block_size) {
        bool has_rows = (num_rows) > 0;

        cb_out0.wait_front(single_block_size * has_rows);
        uint32_t l1_read_addr = cb_out0.get_read_ptr();

        for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            uint32_t total_size = start_column_id + width_size;
            uint32_t write_size = width_size;

            if (total_size > unpadded_X_size) {
                uint32_t padded_size = total_size - unpadded_X_size;
                write_size -= padded_size;
            }

            uint32_t src_offset = l1_read_addr - cb_out0.get_read_ptr();
            noc.async_write(
                cb_out0,
                s,
                write_size,
                {.offset_bytes = src_offset},
                {.page_id = size_2d + k, .offset_bytes = start_column_id});

            noc.async_write_barrier();

            l1_read_addr += width_size;
        }

        cb_out0.pop_front(single_block_size * has_rows);
    };

    const uint32_t width_size = get_arg_val<uint32_t>(1);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_row_id = get_arg_val<uint32_t>(2);
        uint32_t start_column_id = get_arg_val<uint32_t>(3);
        uint32_t single_block_size_row_arg = get_arg_val<uint32_t>(4);
        uint32_t single_block_size_col_arg = get_arg_val<uint32_t>(5);
        uint32_t sub_block_width_size = get_arg_val<uint32_t>(6);
        uint32_t single_sub_block_size_row_arg = get_arg_val<uint32_t>(7);

        for (uint32_t b = 0; b < single_block_size_col_arg; b++) {
            uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            for (uint32_t m = 0; m < width_size; m += sub_block_width_size) {
                uint32_t start_column_id_u = start_column_id + m;
                if (this_block_num_rows > 0) {
                    write_block(
                        this_block_num_rows,
                        start_row_id,
                        start_column_id_u,
                        sub_block_width_size,
                        size_2d,
                        single_sub_block_size_row_arg);
                }
            }
            start_row_id += tile_height;
        }
        size_2d += total_num_rows;
    }
}
