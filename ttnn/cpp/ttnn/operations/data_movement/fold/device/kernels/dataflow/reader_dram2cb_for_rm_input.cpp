// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(2);
    constexpr bool src_stick_size_is_power_of_two = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t src_log2_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t aligned_stick_nbytes_l1 = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_input_work = get_arg_val<uint32_t>(1);
    uint32_t end_input_work = get_arg_val<uint32_t>(2);
    const auto s0 =
        get_interleaved_addr_gen<true, src_stick_size_is_power_of_two>(src_addr, stick_nbytes, src_log2_stick_size);

    uint32_t config_l1_addr = get_read_ptr(config_cb_id);
    volatile tt_l1_ptr uint32_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_l1_addr);

    DPRINT << "aligned_stick_nbytes_l1: " << aligned_stick_nbytes_l1 << ENDL();
    for (uint32_t input_idx = start_input_work; input_idx < end_input_work; input_idx++) {
        // Each mapping takes 3 entries: src_idx, dst_idx, is_padding
        uint32_t mapping_offset = (input_idx - start_input_work) * 2;
        uint32_t src_index = config_data[mapping_offset];

        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_orig = l1_write_addr;
        for (uint32_t i = 0; i < stride_w; i++) {
            uint64_t src_noc_addr = get_noc_addr(src_index, s0);
            noc_async_read(src_noc_addr, l1_write_addr, stick_nbytes);
            src_index++;
            l1_write_addr += aligned_stick_nbytes_l1;
        }
        // Read from NOC and write to L1
        noc_async_read_barrier();
        print_bf16_pages(l1_write_addr_orig, stick_nbytes / 2, stride_w);
        cb_push_back(cb_id_in0, 1);
    }
}
