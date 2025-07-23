// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(3);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(4);
    uint32_t padded_stick_size = get_arg_val<uint32_t>(5);
    uint32_t unpadded_stick_size = get_arg_val<uint32_t>(6);
    uint32_t num_dims = get_arg_val<uint32_t>(7);
    uint32_t misalignment = get_arg_val<uint32_t>(8);

    // Get addresses for arrays stored in common args
    volatile tt_l1_ptr uint32_t* num_unpadded_sticks = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(9));

    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(get_arg_addr(10));
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;

    constexpr auto tensor_args = TensorAccessorArgs<0>();
    uint32_t read_size = unpadded_stick_size + misalignment;

    const auto s0 = TensorAccessor(tensor_args, src_addr, padded_stick_size);

    constexpr uint32_t cb_id_in0 = 0;

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_reserve_back(cb_id_in0, num_read_per_barrier);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0, misalignment);
            noc_async_read(src_noc_addr, l1_write_addr, read_size);
            l1_write_addr += padded_stick_size;
            src_stick_id = update_stick_id(src_stick_id, num_dims, num_unpadded_sticks, num_padded_sticks, id_per_dim);
            sticks_read++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_read_per_barrier);
    }
}
