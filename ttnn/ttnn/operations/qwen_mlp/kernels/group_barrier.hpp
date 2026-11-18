// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/noc_semaphore.h"
#include "named_args.hpp"

inline void signal_group_written(const Noc& noc, Semaphore<>& sem) {
    noc.async_write_barrier();
    sem.up(noc, RT_ARG("barrier_x"), RT_ARG("barrier_y"), 1);
}

inline void wait_group_written(Semaphore<>& sem) { sem.wait_min(CT_ARG("barrier_group_size")); }
