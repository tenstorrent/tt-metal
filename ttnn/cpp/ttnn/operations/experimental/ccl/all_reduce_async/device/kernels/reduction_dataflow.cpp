// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    /*
        1. Wait for signal (1)
        2. Clear signal
        4. Push back on interleaved all gather CB
        5. compute wait front on all gather cb
    */
    uint32_t signal_semaphore_addr = get_semaphore(get_compile_time_arg_val(0));

    volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);

    DPRINT << "Wait \n";
    // 1. Wait for signal
    noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
    DPRINT << " Wait Over \n";
    // 2. Synchronization
}
