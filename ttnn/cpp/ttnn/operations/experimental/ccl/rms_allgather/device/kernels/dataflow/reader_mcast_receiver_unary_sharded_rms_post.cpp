// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ex_global = get_compile_time_arg_val(1);  // [E[x], E[X^2]] global to all cores
    uint32_t reduce_sender_semaphore_addr = get_semaphore(semaphore_id);

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    // inc mcast sender
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
    // inc remote sem
    cb_reserve_back(cb_ex_global, 1);
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
    cb_push_back(cb_ex_global, 1);
}
