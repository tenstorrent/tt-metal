// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
// #include "debug/debug.h"

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
    if constexpr (out_nsticks_per_core == 0) {
        // No output sticks to write, so return early.
        return;
    }
    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    uint32_t l1_write_addr = get_write_ptr(out_cb_id);
    if (!is_reader) {
        l1_write_addr += out_nsticks_per_core * stick_nbytes;
    }

    uint32_t config_l1_addr = get_read_ptr(config_cb_id);
    volatile tt_l1_ptr uint16_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);

    uint32_t reader_idx = 0;
    if constexpr (!is_reader) {
        reader_idx = elem_per_core_reader * 4;
    }
    // cb_reserve_back(out_cb_id, out_nsticks_per_core);

    for (uint32_t row_begin = 0; row_begin < elem_per_core_reader; ++row_begin) {
        uint16_t cores = config_data[reader_idx++];
        uint16_t corey = cores & 0xFF;
        uint16_t corex = cores >> 8;
        uint16_t offset_start = config_data[reader_idx++];
        uint16_t offset_end = config_data[reader_idx++];
        reader_idx++;  // pad
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
    // cb_push_back(out_cb_id, out_nsticks_per_core);
}
