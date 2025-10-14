// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb = get_compile_time_arg_val(1);
    constexpr uint32_t pixel_size = get_compile_time_arg_val(2);
    constexpr uint32_t aligned_pixel_size = get_compile_time_arg_val(3);
    constexpr uint32_t aligned_dst_pixel_size = get_compile_time_arg_val(4);
    constexpr uint32_t aligned_chunk_size = get_compile_time_arg_val(5);
    constexpr uint32_t aligned_row_size = get_compile_time_arg_val(6);
    constexpr uint32_t stride_h = get_compile_time_arg_val(7);
    constexpr uint32_t stride_w = get_compile_time_arg_val(8);
    constexpr uint32_t num_dst_rows = get_compile_time_arg_val(9);
    constexpr uint32_t num_dst_cols = get_compile_time_arg_val(10);
    constexpr uint32_t dst_row_offset = get_compile_time_arg_val(11);
    constexpr uint32_t is_reader = get_compile_time_arg_val(12);

    constexpr uint32_t dst_row_size = num_dst_cols * aligned_dst_pixel_size;
    constexpr uint32_t cols_per_core = num_dst_cols / 2;
    constexpr uint32_t process_cols = cols_per_core + ((num_dst_cols % 2) & is_reader);
    constexpr uint32_t core_col_offset = is_reader ? 0 : aligned_chunk_size;
    constexpr uint32_t core_dst_offset = is_reader ? 0 : aligned_dst_pixel_size;

    constexpr bool is_aligned = (pixel_size == aligned_pixel_size);
    constexpr uint32_t read_size = is_aligned ? aligned_chunk_size : pixel_size;
    constexpr uint32_t num_pixels = pixel_size / 2;
    constexpr uint32_t pixel_skip_u16 = (aligned_pixel_size - pixel_size) / 2;

    uint64_t src_noc_addr = get_noc_addr(get_read_ptr(src_cb));
    const uint32_t dst_addr_base = get_write_ptr(dst_cb);
    uint32_t src_addr_base = get_read_ptr(src_cb);

    if (is_aligned) {
        noc_async_read_one_packet_set_state(src_noc_addr, read_size);
    }

    for (uint32_t row = 0; row < num_dst_rows; ++row) {
        uint64_t src_col_offset = core_col_offset;
        uint32_t dst_addr = dst_addr_base + (row * dst_row_size) + core_dst_offset;

        for (uint32_t col = 0; col < process_cols; ++col) {
            uint32_t dst_pixel_addr = dst_addr;
            for (uint32_t h = 0; h < stride_h; ++h) {
                const uint32_t h_offset = h * aligned_row_size;
                if constexpr (is_aligned) {
                    noc_async_read_one_packet_with_state<true>(
                        src_noc_addr + src_col_offset + h_offset, dst_pixel_addr);
                    dst_pixel_addr += aligned_chunk_size;
                } else {
                    volatile tt_l1_ptr uint16_t* src_h_addr =
                        (volatile tt_l1_ptr uint16_t*)(src_addr_base + src_col_offset + h_offset);
                    volatile tt_l1_ptr uint16_t* dst_h_addr = (volatile tt_l1_ptr uint16_t*)dst_pixel_addr;

                    for (uint32_t w = 0; w < stride_w; ++w) {
                        for (uint32_t i = 0; i < num_pixels; ++i) {
                            *dst_h_addr++ = *src_h_addr++;
                        }
                        src_h_addr += pixel_skip_u16;
                    }
                    dst_pixel_addr += pixel_size * stride_w;
                }
            }
            src_col_offset += aligned_chunk_size * 2;
            dst_addr += aligned_dst_pixel_size * 2;
        }
        src_noc_addr += dst_row_offset;
        src_addr_base += dst_row_offset;
    }

    noc_async_read_barrier();
}
