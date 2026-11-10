// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage hand-off, producer side: once every write this RISC has issued so far has landed, increment semaphore
// A::sem on each of the cores listed in the runtime args. Placed after a producer block on the RISC that does its
// writes. (Blocks run all init()s, then all run()s, then all teardown()s, so the barrier here is what makes "after
// the previous block's run()" mean "after its writes landed".)
//
// Compile-time values (A::): sem (semaphore id, same on every core of the program)
// Runtime args: num_targets, then noc_x, noc_y per target

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::sync {

template <typename A>
struct Signal {
    void init() {}

    void run() {
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        const uint32_t sem_addr = get_semaphore(A::sem);
        const uint32_t n = rt<A>(0);
        for (uint32_t i = 0; i < n; i++) {
            const uint64_t addr = get_noc_addr(rt<A>(1 + 2 * i), rt<A>(2 + 2 * i), sem_addr);
            noc_semaphore_inc(addr, 1);
        }
        noc_async_atomic_barrier();
    }

    void teardown() {}
};

}  // namespace lego::sync
