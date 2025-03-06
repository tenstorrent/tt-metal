// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t receiver_semaphore = get_semaphore(get_compile_time_arg_val(0));
    uint32_t sender_semaphore = get_semaphore(get_compile_time_arg_val(1));

    constexpr uint32_t noc_start_x = get_compile_time_arg_val(2);
    constexpr uint32_t noc_start_y = get_compile_time_arg_val(3);
    constexpr uint32_t noc_end_x = get_compile_time_arg_val(4);
    constexpr uint32_t noc_end_y = get_compile_time_arg_val(5);

    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt_final = get_compile_time_arg_val(7);
    constexpr uint32_t num_dests = get_compile_time_arg_val(8);

    constexpr uint32_t final_values_cb_index = tt::CBIndex::c_26;
    constexpr uint32_t final_indices_cb_index = tt::CBIndex::c_27;

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore);
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore);

    uint64_t mcast_receiver_semaphore_noc_addr =
        get_noc_multicast_addr(noc_start_x, noc_start_y, noc_end_x, noc_end_y, receiver_semaphore);

    for (uint32_t i = 0; i < Ht; ++i) {
        // Look for space in buffer
        cb_reserve_back(final_values_cb_index, Wt_final);
        cb_reserve_back(final_indices_cb_index, Wt_final);

        // Data is unsent so label the sender semaphore as INVALID
        noc_semaphore_set(sender_semaphore_addr, INVALID);

        // Set the receiver semaphore to VALID to allow the sender to write
        noc_semaphore_set(receiver_semaphore_addr, VALID);

        // Update the multicast address for the receiver semaphore, to allow the senders to write
        noc_semaphore_set_multicast(receiver_semaphore, mcast_receiver_semaphore_noc_addr, num_dests);
        noc_semaphore_wait(sender_semaphore_addr, Wt_final);

        cb_push_back(final_values_cb_index, Wt_final);
        cb_push_back(final_indices_cb_index, Wt_final);
    }
}
