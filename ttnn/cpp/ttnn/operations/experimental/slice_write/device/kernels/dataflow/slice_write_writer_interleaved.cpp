// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_stick_size = get_arg_val<uint32_t>(1);
    const uint32_t input_stick_size = get_arg_val<uint32_t>(2);
    const uint32_t stick_size_offset = get_arg_val<uint32_t>(3);
    const uint32_t num_dims = get_arg_val<uint32_t>(4);
    const uint32_t start_id = get_arg_val<uint32_t>(5);
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(6);
    const uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(7);
    const uint32_t num_read_per_barrier = get_arg_val<uint32_t>(8);
#ifdef DEBUG
    DPRINT << "dst_addr: " << dst_addr << ENDL();
    DPRINT << "output_stick_size: " << output_stick_size << ENDL();
    DPRINT << "input_stick_size: " << input_stick_size << ENDL();
    DPRINT << "stick_size_offset: " << stick_size_offset << ENDL();
    DPRINT << "num_dims: " << num_dims << ENDL();
    DPRINT << "start_id: " << start_id << ENDL();
    DPRINT << "num_sticks_per_core: " << num_sticks_per_core << ENDL();
    DPRINT << "num_sticks_per_core_read: " << num_sticks_per_core_read << ENDL();
    DPRINT << "num_read_per_barrier: " << num_read_per_barrier << ENDL();
#endif
    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(get_arg_addr(9));
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr bool dst0_is_dram = get_compile_time_arg_val(1) == 1;

    const InterleavedAddrGen<dst0_is_dram> s0 = {.bank_base_address = dst_addr, .page_size = output_stick_size};
    const uint32_t noc_write_size = std::min(output_stick_size, input_stick_size);
    uint32_t dst_stick_id = start_id;
    uint32_t sticks_read = 0;
#ifdef DEBUG
    uint32_t base_src_l1_addr = get_read_ptr(cb_id_out0);
#endif
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_wait_front(cb_id_out0, num_read_per_barrier);
        uint32_t src_buffer_l1_addr = get_read_ptr(cb_id_out0);

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, s0);
            noc_async_write(src_buffer_l1_addr, dst_noc_addr, noc_write_size);
#ifdef DEBUG
            DPRINT << "SRC L1 : " << src_buffer_l1_addr - base_src_l1_addr << " Dst Stick ID " << dst_stick_id
                   << " Coord " << id_per_dim[0] << ", " << id_per_dim[1] << ", " << id_per_dim[2] << ", "
                   << id_per_dim[3] << ENDL();
#endif
            src_buffer_l1_addr += stick_size_offset;
            dst_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    dst_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_read_per_barrier);
    }
}
