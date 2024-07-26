// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t padded_stick_size        = get_arg_val<uint32_t>(1);
    const uint32_t unpadded_stick_size      = get_arg_val<uint32_t>(2);
    const uint32_t stick_size_offset        = get_arg_val<uint32_t>(3);
    const uint32_t num_dims                 = get_arg_val<uint32_t>(4);
    const uint32_t start_id                 = get_arg_val<uint32_t>(5);
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(6);
    const uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(7);
    const uint32_t num_read_per_barrier = get_arg_val<uint32_t>(8);

    tt_l1_ptr uint32_t * num_unpadded_sticks = (tt_l1_ptr uint32_t*)(get_arg_addr(9));
    volatile tt_l1_ptr uint32_t * num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t * id_per_dim = num_padded_sticks + num_dims;

    constexpr bool src0_is_dram          = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGen<src0_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = padded_stick_size
    };

    constexpr uint32_t cb_id_in0 = 0;

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_reserve_back(cb_id_in0, num_read_per_barrier);
        uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0);
            noc_async_read(src_noc_addr, src_buffer_l1_addr, unpadded_stick_size);
            src_buffer_l1_addr += stick_size_offset;
            src_stick_id++;
            for(uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_read_per_barrier);
    }
}
