// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage hand-off, consumer side: wait until the local semaphore A::sem reaches A::count (the number of producer
// Signal blocks targeting this core). Placed before a consumer block on the RISC that starts the consumer's reads.
// Program semaphores are re-initialised at every launch, so no reset is needed.
//
// Compile-time values (A::): sem count

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"

namespace lego::sync {

template <typename A>
struct Wait {
    void init() {}

    void run() {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A::sem));
        noc_semaphore_wait_min(sem, A::count);
    }

    void teardown() {}
};

}  // namespace lego::sync
