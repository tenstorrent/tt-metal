// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "dataflow_api.h"

void kernel_main() {
    uint32_t arg_idx = 0;
    volatile uint32_t* sem0 = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    volatile uint32_t* sem1 = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    const uint32_t receiver_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receiver_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t core0_start_sem = get_arg_val<uint32_t>(arg_idx++);
    noc_semaphore_wait(sem0, 1);
    noc_semaphore_wait(sem1, 1);
    uint64_t core0_start_sem_noc_addr = get_noc_addr(receiver_x, receiver_y, core0_start_sem);

    noc_semaphore_inc(core0_start_sem_noc_addr, 1);
}
