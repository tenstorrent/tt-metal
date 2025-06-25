// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_num_reduction_tiles = get_compile_time_arg_val(1);

    // runtime args
    size_t arg_idx = 0;
    const uint32_t has_work = get_arg_val<uint32_t>(arg_idx++);
    if (has_work == 0) {
        return;
    }

    const uint32_t signal_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    DPRINT << "reduction_receiver out_ready_sem_wait_value: " << out_ready_sem_wait_value << "\n";
    DPRINT << "reduction_receiver before out_ready_sem_value: "
           << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) << "\n";
    // while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) != out_ready_sem_wait_value);
    DPRINT << "reduction_receiver afterward out_ready_sem_value: "
           << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) << "\n";

    /*
    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    // 1. Wait for signal from All-Gather worker
    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
    noc_semaphore_set(signal_semaphore_addr_ptr, 0);
    */

    // 2. Signal compute kernel to start processing
    cb_push_back(cb_id, total_num_reduction_tiles);
    DPRINT << "to reset global semaphore\n";
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) = 0;
    DPRINT << "reduction_receiver done \n";
}
