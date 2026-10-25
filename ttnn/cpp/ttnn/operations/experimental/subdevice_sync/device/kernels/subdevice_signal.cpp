// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// subdevice_signal: one core on the PRODUCER's sub-device atomically adds `value` to a GlobalSemaphore word in the
// L1 of `target` (the waiter core). Runs only after every earlier program on this sub-device has completed (its go
// signal waits for them), so the increment is a "producer done" token.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t sem_addr = get_arg_val<uint32_t>(0);
    const uint32_t target_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t target_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t value = get_arg_val<uint32_t>(3);
    noc_semaphore_inc(get_noc_addr(target_noc_x, target_noc_y, sem_addr), value);
    // Non-posted atomic: wait for the ack so the increment has landed before this program reports completion.
    noc_async_atomic_barrier();
}
