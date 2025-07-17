// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(2);
    constexpr bool dst_stick_size_is_power_of_two = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t dst_log2_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t aligned_stick_nbytes_dram = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_input_work = get_arg_val<uint32_t>(1);
    uint32_t end_input_work = get_arg_val<uint32_t>(2);
    const auto s_out =
        get_interleaved_addr_gen<true, dst_stick_size_is_power_of_two>(dst_addr, stick_nbytes, dst_log2_stick_size);
    uint32_t config_l1_addr = get_read_ptr(config_cb_id);
    volatile tt_l1_ptr uint32_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_l1_addr);

    for (uint32_t input_idx = start_input_work; input_idx < end_input_work; input_idx++) {
        // Each mapping takes 2 entries: src_idx, dst_idx
        uint32_t mapping_offset = (input_idx - start_input_work) * 2;
        uint32_t src_index = config_data[mapping_offset];
        uint32_t dst_index = config_data[mapping_offset + 1];

        cb_wait_front(cb_id_in0, 1);
        uint32_t l1_addr = get_read_ptr(cb_id_in0);
        for (uint32_t i = 0; i < stride_w; i++) {
            uint64_t dst_noc_addr = get_noc_addr(dst_index, s_out);
            noc_async_write(l1_addr, dst_noc_addr, stick_nbytes);
            dst_index++;
            l1_addr += aligned_stick_nbytes_dram;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_in0, 1);
    }
}
