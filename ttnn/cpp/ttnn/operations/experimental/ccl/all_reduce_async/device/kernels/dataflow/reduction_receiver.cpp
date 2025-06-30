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
    volatile tt_l1_ptr uint32_t* out_ready_sema =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);

    // 1. Wait for signal from All-Gather worker
    noc_semaphore_wait(out_ready_sema, out_ready_sem_wait_value);

    // 2. Signal compute kernel to start processing
    cb_push_back(cb_id, total_num_reduction_tiles);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
}
