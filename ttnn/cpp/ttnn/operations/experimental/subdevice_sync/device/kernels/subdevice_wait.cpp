// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// subdevice_wait: one core on the CONSUMER's sub-device spins on a GlobalSemaphore word in its own L1 until it is
// >= `value`, then atomically subtracts `value` (a counting semaphore: increments that race with the consume are
// never lost, so the semaphore returns to its initial value after every balanced signal/wait pair and each trace
// replay starts clean). The next program on this sub-device starts only after this one completes.
//
// timeout_cycles != 0 (debug only): give up after that many AICLK cycles WITHOUT consuming; the consumer then runs
// unordered. Use it to turn a would-be hang (mis-ordered issue) into detectable ordering violations.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
FORCE_INLINE uint64_t wall_clock() {
    volatile uint32_t tt_reg_ptr* clock_lo = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint32_t tt_reg_ptr* clock_hi = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint32_t lo = clock_lo[0];  // reading LO latches HI
    return lo | (static_cast<uint64_t>(clock_hi[0]) << 32);
}
}  // namespace

void kernel_main() {
    const uint32_t sem_addr = get_arg_val<uint32_t>(0);
    const uint32_t value = get_arg_val<uint32_t>(1);
    const uint32_t timeout_cycles = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    if (timeout_cycles == 0) {
        noc_semaphore_wait_min(sem, value);
    } else {
        const uint64_t start = wall_clock();
        while (true) {
            invalidate_l1_cache();
            if (*sem >= value) {
                break;
            }
            if (wall_clock() - start > timeout_cycles) {
                return;  // timed out: leave the semaphore untouched
            }
        }
    }
    // Consume: atomic add of -value (NOC atomics wrap mod 2^32) to our own L1 word.
    noc_semaphore_inc(get_noc_addr(sem_addr), 0u - value);
    noc_async_atomic_barrier();
}
