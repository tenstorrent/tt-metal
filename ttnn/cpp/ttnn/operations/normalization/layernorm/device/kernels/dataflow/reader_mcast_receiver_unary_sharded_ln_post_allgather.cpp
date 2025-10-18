// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// split REDUCE across cores
void kernel_main() {
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t block_h = get_compile_time_arg_val(3);
    constexpr bool rms_norm = get_compile_time_arg_val(15) == 1;

    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // [E[x], E[X^2]] global to all cores

    constexpr uint32_t stats_tiles = rms_norm ? 1 : 2;

    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    // inc mcast sender
    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
    // inc remote sem
    cb_reserve_back(cb_ex_global, stats_tiles * block_h);
    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
    cb_push_back(cb_ex_global, stats_tiles * block_h);
}
