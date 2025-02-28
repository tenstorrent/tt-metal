// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);
    uint32_t scale_h = get_arg_val<uint32_t>(3);
    uint32_t scale_w = get_arg_val<uint32_t>(4);
    uint32_t height = get_arg_val<uint32_t>(5);
    uint32_t width = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);

    constexpr bool dst0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool dst_stick_size_is_pow2 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);

    const auto s0 = get_interleaved_addr_gen<dst0_is_dram, dst_stick_size_is_pow2>(
        dst_addr, stick_size, dst_log_base_2_of_page_size);

    uint32_t scale = scale_h * scale_w;
    uint32_t in_width = width / scale_w;
    uint32_t in_height = height / scale_h;
    // reader copied the data from DRAM to CB buffer.
    // writer copy the data from CB buffer to DRAM.
    for (uint32_t i = 0; i < num_sticks; ++i) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t curr_index = i % (in_width * in_height);
        uint32_t curr_batch = i / (in_width * in_height);
        uint32_t x = curr_index / in_width;
        uint32_t y = curr_index % in_width;
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        // calculate the start index where writer will start writing the data.
        // total --> scale_h * scale_w times data will be written to the DRAM.
        // offset calcutes the relative position of the data in the stick.
        uint32_t start_index = curr_batch * width * height + (scale_h * x) * width + scale_w * y;
        for (uint32_t j = 0; j < scale_h; j++) {
            for (size_t k = 0; k < scale_w; k++) {
                uint64_t offset = j * width + k;
                uint64_t dst_noc_addr_1 = get_noc_addr((start_index + offset), s0);
                noc_async_write(l1_read_addr, dst_noc_addr_1, stick_size);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
