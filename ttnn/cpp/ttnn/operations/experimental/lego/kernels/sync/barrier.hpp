// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Barrier over a group of cores of one program: every member first waits for its own outstanding writes to land,
// then increments the leader's arrive semaphore; the leader waits for all A::num_members arrivals and increments
// every member's release semaphore (itself included); every member then waits for its release. After the barrier,
// every write any member issued before it is visible to every member.
//
// CT (A::): arrive_sem release_sem num_members is_leader
// RT: leader_noc_x leader_noc_y, then (leader only) noc_x noc_y per member

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::sync {

template <typename A>
struct Barrier {
    void init() {}

    void run() {
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        const uint32_t arrive = get_semaphore(A::arrive_sem);
        const uint32_t release = get_semaphore(A::release_sem);
        noc_semaphore_inc(get_noc_addr(rt<A>(0), rt<A>(1), arrive), 1);
        if constexpr (A::is_leader) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive), A::num_members);
            for (uint32_t i = 0; i < A::num_members; i++) {
                noc_semaphore_inc(get_noc_addr(rt<A>(2 + 2 * i), rt<A>(3 + 2 * i), release), 1);
            }
        }
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release), 1);
    }

    void teardown() {}
};

}  // namespace lego::sync
