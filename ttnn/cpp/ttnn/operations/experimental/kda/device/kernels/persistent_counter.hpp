// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// The KDA cross-device counters: global semaphores that keep their counts across calls, which peers bump over the
// fabric. A peer may count the next call before this core consumes this call's count, so a count is consumed by
// subtracting it, never by clearing. After a run that stopped mid-call, the host zeroes them
// (reset_persistent_semaphores in relay_affine_prefix.py).
namespace kda_counter {

// Wait until the counter at `address` on this core reaches `count`.
FORCE_INLINE void wait(uint32_t address, uint32_t count) {
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address), count);
}

// Take `count` off the counter at `address` on this core.
FORCE_INLINE void consume(uint32_t address, uint32_t count) {
    noc_semaphore_inc(get_noc_addr(address), 0u - count);
    noc_async_atomic_barrier();
}

FORCE_INLINE void wait_and_consume(uint32_t address, uint32_t count) {
    wait(address, count);
    consume(address, count);
}

}  // namespace kda_counter
