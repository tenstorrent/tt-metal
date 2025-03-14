// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_unpadded_W = get_arg_val<uint32_t>(2);
    const uint32_t num_total_W = get_arg_val<uint32_t>(3);
    const uint32_t num_unpadded_Z = get_arg_val<uint32_t>(4);
    const uint32_t num_total_Z = get_arg_val<uint32_t>(5);
    const uint32_t num_unpadded_Y = get_arg_val<uint32_t>(6);
    const uint32_t num_total_Y = get_arg_val<uint32_t>(7);
    const uint32_t num_unpadded_X = get_arg_val<uint32_t>(8);
    const uint32_t num_total_X = get_arg_val<uint32_t>(9);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(10);
    const uint32_t padded_X_size = get_arg_val<uint32_t>(11);
    const uint32_t padded_X_diff_size = get_arg_val<uint32_t>(12);
    const uint32_t pad_value = get_arg_val<uint32_t>(13);
    const uint32_t dst_buffer_l1_addr = get_arg_val<uint32_t>(14);

    tt_l1_ptr std::uint32_t* dst_buffer = (tt_l1_ptr uint32_t*)(dst_buffer_l1_addr);
    volatile tt_l1_ptr std::uint32_t* dst_pad = (volatile tt_l1_ptr uint32_t*)(dst_buffer_l1_addr + unpadded_X_size);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool src_stick_size_is_pow2 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(3);
    constexpr bool dst_stick_size_is_pow2 = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(5);

    const auto s0 = get_interleaved_addr_gen<src0_is_dram, src_stick_size_is_pow2>(
        src_addr, unpadded_X_size, src_log_base_2_of_page_size);
    const auto s1 = get_interleaved_addr_gen<dst_is_dram, dst_stick_size_is_pow2>(
        dst_addr, padded_X_size, dst_log_base_2_of_page_size);

    uint32_t padded_X_size_by_4 = padded_X_size >> 2;
    uint32_t padded_X_diff_size_by_4 = padded_X_diff_size >> 2;

    uint32_t src_stick_id = 0;
    uint32_t dst_stick_id = 0;
    for (uint32_t w = 0; w < num_total_W; w++) {
        for (uint32_t z = 0; z < num_total_Z; z++) {
            for (uint32_t y = 0; y < num_total_Y; y++) {
                if (y >= num_unpadded_Y || z >= num_unpadded_Z || w >= num_unpadded_W) {
                    // pad the tile with pad_value
                    for (uint32_t i = 0; i < padded_X_size_by_4; i++) {
                        dst_buffer[i] = pad_value;
                    }
                } else {
                    uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0);
                    noc_async_read(src_noc_addr, dst_buffer_l1_addr, unpadded_X_size);
                    // Pad Columns first
                    for (uint32_t i = 0; i < padded_X_diff_size_by_4; i++) {
                        dst_pad[i] = pad_value;
                    }
                    noc_async_read_barrier();
                    src_stick_id++;
                }
                uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, s1);
                noc_async_write(dst_buffer_l1_addr, dst_noc_addr, padded_X_size);
                noc_async_write_barrier();
                dst_stick_id++;
            }
        }
    }
}
