// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t sem_wait_val = get_compile_time_arg_val(0);

    // runtime args
    size_t arg_idx = 0;
    const uint32_t signal_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // 1. Wait for signal
    {
        DeviceZoneScopedN("waiting");
        noc_semaphore_wait_min(signal_semaphore_addr_ptr, sem_wait_val);
        noc_semaphore_set(signal_semaphore_addr_ptr, 0);
    }
}
