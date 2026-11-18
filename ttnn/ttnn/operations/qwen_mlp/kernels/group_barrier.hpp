// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/noc_semaphore.h"
#include "named_args.hpp"

// Every member of the group calls this once its DRAM writes are issued. Only the injector waits for the whole group.
inline void group_barrier(const Noc& noc, Semaphore<>& sem, bool is_injector) {
    noc.async_write_barrier();
    sem.up(noc, RT_ARG("barrier_x"), RT_ARG("barrier_y"), 1);
    if (is_injector) {
        sem.wait_min(CT_ARG("barrier_group_size"));
    }
}
