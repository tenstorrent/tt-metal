// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This should ideally be merged with `ccl_send_reader` when we are able to support compile time args
//       that don't require macros to function

#include "dataflow_api.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
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

    size_t arg_idx = 0;
    const size_t wait_signal_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* l1_wait_signal_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wait_signal_sem_addr);

    // 1. Wait for signal
    noc_semaphore_wait(l1_wait_signal_sem, 1);
    noc_semaphoer_reset(l1_wait_signal_sem, 0);

    // 2. Synchronization
}
