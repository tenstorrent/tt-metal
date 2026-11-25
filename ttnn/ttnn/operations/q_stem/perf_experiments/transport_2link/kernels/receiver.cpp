// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// q_stem transport bench: destination core (stands in for stage 3's arrival waits). Waits until every source's
// arrival counter holds all the tiles it sends this core, then re-arms the counter (trace-safe), so the program ends
// on a chip when its last remote row has landed.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t addr = get_arg_val<uint32_t>(1 + 2 * i);
        const uint32_t expect = get_arg_val<uint32_t>(2 + 2 * i);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr), expect);
        noc_semaphore_inc(get_noc_addr(addr), 0u - expect);
    }
    noc_async_atomic_barrier();
}
