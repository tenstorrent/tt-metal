// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ckernel.h"

void kernel_main() {
    DPRINT << "Kernel = worker_receiver" << ENDL();
    return;
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t sem_wait_val = get_compile_time_arg_val(0);
    DPRINT << "sem_wait_val: " << sem_wait_val << ENDL();

    // runtime args
    size_t arg_idx = 0;
    const uint32_t signal_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_id = get_arg_val<uint32_t>(
        arg_idx++);  // core id, corresponds to the id of which device it expect data from, will be reset later
    DPRINT << "core_id: " << core_id << ENDL();

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // 1. Wait for signal
    {
        DeviceZoneScopedN("data waiting");
        uint64_t t1 = ckernel::read_wall_clock();
        noc_semaphore_wait_min(signal_semaphore_addr_ptr, sem_wait_val);
        noc_semaphore_set(signal_semaphore_addr_ptr, 0);
        uint64_t t2 = ckernel::read_wall_clock();
        DPRINT << "time taken(in us): " << (t2 - t1) << ENDL();
    }
}
