// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t src_shard_cb = get_compile_time_arg_val(0);
    constexpr uint32_t dst_shard_cb = get_compile_time_arg_val(1);

    const uint32_t pixel_size = get_arg_val<uint32_t>(0);
    const uint32_t aligned_pixel_size = get_arg_val<uint32_t>(1);
    const uint32_t aligned_dst_pixel_size = get_arg_val<uint32_t>(2);
    const uint32_t num_src_pixels = get_arg_val<uint32_t>(3);
    const uint32_t num_dst_pixels = get_arg_val<uint32_t>(4);

    const uint32_t aligned_chunk_size = get_arg_val<uint32_t>(5);
    const uint32_t aligned_row_size = get_arg_val<uint32_t>(6);

    const uint32_t stride_h = get_arg_val<uint32_t>(7);
    const uint32_t stride_w = get_arg_val<uint32_t>(8);

    const uint32_t num_dst_rows = get_arg_val<uint32_t>(9);
    const uint32_t num_dst_cols = get_arg_val<uint32_t>(10);
    const uint32_t dst_row_offset = get_arg_val<uint32_t>(11);

    auto copy_next_dst_pixel = [&](uint64_t src_noc_address, uint32_t dst_l1_addr) -> void {
        for (uint32_t row = 0, row_offset = 0; row < stride_h; ++row, row_offset += aligned_row_size) {
            for (uint32_t col = 0, col_offset = 0; col < stride_w; ++col, col_offset += aligned_pixel_size) {
                uint32_t offset = col_offset + row_offset;
                noc_async_read(src_noc_address + offset, dst_l1_addr, pixel_size);
                dst_l1_addr += pixel_size;
            }
        }
    };

    uint64_t src_noc_addr = get_noc_addr(get_read_ptr(src_shard_cb));
    uint32_t dst_addr = get_write_ptr(dst_shard_cb);

    for (uint32_t i = 0; i < num_dst_rows; ++i) {
        uint64_t src_col_offset = 0;
        for (uint32_t j = 0; j < num_dst_cols; ++j) {
            copy_next_dst_pixel(src_noc_addr + src_col_offset, dst_addr);
            src_col_offset += aligned_chunk_size;
            dst_addr += aligned_dst_pixel_size;
        }
        src_noc_addr += dst_row_offset;
        noc_async_read_barrier();
    }
}
