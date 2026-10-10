// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x-download probe, receiver (BRISC): waits until every reader has finished sending (DONE reaches N_READERS).
// CT: 0 DONE_SEM, 1 N_READERS
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t done_sem = get_compile_time_arg_val(0);
    constexpr uint32_t n_readers = get_compile_time_arg_val(1);
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_sem)), n_readers);
}
