// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t stick_nbytes = get_arg_val<uint32_t>(0);
    uint32_t in_image_rows_per_core = get_arg_val<uint32_t>(1);
    uint32_t scale_h = get_arg_val<uint32_t>(2);
    uint32_t scale_w = get_arg_val<uint32_t>(3);
    uint32_t in_w = get_arg_val<uint32_t>(4);
    uint32_t out_w = get_arg_val<uint32_t>(5);

    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t is_reader = get_compile_time_arg_val(2);

    uint32_t in_image_row_nbytes = in_w * stick_nbytes;
    uint32_t out_image_row_nbytes = out_w * stick_nbytes;
    uint32_t reader_image_rows_per_core = (in_image_rows_per_core + is_reader) / 2;
    uint32_t writer_image_rows_per_core = in_image_rows_per_core / 2;
    uint32_t image_row_begin = is_reader ? 0 : reader_image_rows_per_core;
    uint32_t image_row_end = is_reader ? reader_image_rows_per_core : in_image_rows_per_core;
    uint32_t l1_read_addr = get_read_ptr(in_cb_id) + image_row_begin * in_image_row_nbytes;
    uint32_t l1_write_addr = get_write_ptr(out_cb_id) + image_row_begin * scale_h * out_image_row_nbytes;

    cb_reserve_back(out_cb_id, out_w);

    // assuming shard begins with a new row. TODO: generalize?
    for (uint32_t image_row = image_row_begin; image_row < image_row_end; ++image_row) {
        uint32_t l1_write_addr_image_row_start = l1_write_addr;
        for (uint32_t i = 0; i < in_w; ++i) {
            // replicate stick scale_w times.
            for (uint32_t sw = 0; sw < scale_w; ++sw) {
                // replicate stick scale_w times.
                if constexpr (is_reader) {
                    uint64_t src_noc_addr = get_noc_addr(l1_read_addr);
                    noc_async_read(src_noc_addr, l1_write_addr, stick_nbytes);
                } else {
                    uint64_t dst_noc_addr = get_noc_addr(l1_write_addr);
                    noc_async_write(l1_read_addr, dst_noc_addr, stick_nbytes);
                }
                l1_write_addr += stick_nbytes;
            }
            l1_read_addr += stick_nbytes;
        }

        // Duplicate the whole image row in one shot
        if constexpr (is_reader) {
            uint64_t src_noc_addr = get_noc_addr(l1_write_addr_image_row_start);
            noc_async_read(src_noc_addr, l1_write_addr, out_image_row_nbytes);
        } else {
            uint64_t dst_noc_addr = get_noc_addr(l1_write_addr);
            noc_async_write(l1_write_addr_image_row_start, dst_noc_addr, out_image_row_nbytes);
        }
        l1_write_addr += out_image_row_nbytes;
    }

    cb_push_back(out_cb_id, out_w);

    noc_async_write_barrier();
    noc_async_read_barrier();
}
