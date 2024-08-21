// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // in0 mcast args
    const uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(1);

    // COMPILE TIME ARGS
    // in0 block args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    // in0/in1 common args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    // batch args
    constexpr uint32_t batch = get_compile_time_arg_val(4);

    constexpr uint32_t cb_id_in0 = 0;

    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);

    const uint64_t in0_mcast_sender_semaphore_noc_addr =
        get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_mcast_sender_semaphore_addr);

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t block = 0; block < num_blocks; ++block) {
            // Operand 0
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);

            // Set in0 semaphore value to INVALID
            noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

            // Atomic increment source core counter
            noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);

            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);

            cb_push_back(cb_id_in0, in0_block_num_tiles);
        }
    }
}
