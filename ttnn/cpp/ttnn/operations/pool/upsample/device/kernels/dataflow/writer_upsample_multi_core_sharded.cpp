// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t is_reader = get_compile_time_arg_val(2);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(4);
    constexpr uint32_t in_nsticks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t scale_h = get_compile_time_arg_val(6);
    constexpr uint32_t scale_w = get_compile_time_arg_val(7);
    constexpr uint32_t elem_per_core = get_compile_time_arg_val(8);

    constexpr uint32_t elem_per_core_reader = elem_per_core / 2;

    constexpr uint32_t out_nsticks_per_core =
        ((in_nsticks_per_core * scale_h + 1) / 2) *
        scale_w;  // divided by 2 because each core has 2 readers which get near equal number of output sticks

    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    uint32_t l1_write_addr = get_write_ptr(out_cb_id);
    if (!is_reader) {
        l1_write_addr += out_nsticks_per_core * stick_nbytes;
    }

    uint32_t config_l1_addr = get_read_ptr(config_cb_id);
    // Interpreted as a vector of 32bit elements to lessen the number of l1 reads
    volatile tt_l1_ptr uint32_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_l1_addr);

    uint32_t reader_idx = 0;
    if constexpr (!is_reader) {
        // Skip first half of config vector, 2 is for number of 32bit elements per entry
        // | core.x |  core.y  | stick_offset_start | stick_offset_end |  (each 16 bits)
        reader_idx = elem_per_core_reader * 2;
    }

    for (uint32_t row_begin = 0; row_begin < elem_per_core_reader; ++row_begin) {
        const uint32_t cores = config_data[reader_idx++];  // Extract the core coordinates
        const uint16_t corex = cores & 0xFFFF;
        const uint16_t corey = cores >> 16;

        const uint32_t offset_info = config_data[reader_idx++];  // Extract offset start and offset end
        const uint16_t offset_start = offset_info & 0xFFFF;      // Little endian RISCV
        const uint16_t offset_end = offset_info >> 16;

        for (uint32_t offset = offset_start; offset <= offset_end; offset++) {
            uint64_t src_remote_addr = get_noc_addr(corex, corey, l1_read_addr + offset * stick_nbytes);
            // replicate stick scale_w times.
            for (uint32_t sw = 0; sw < scale_w; sw++) {
                noc_async_read(src_remote_addr, l1_write_addr, stick_nbytes);
                l1_write_addr += stick_nbytes;
            }
        }
    }

    noc_async_read_barrier();
}
