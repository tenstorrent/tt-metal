// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute for the gated-delta prefill-then-query op — PLACEHOLDER CONSUMER.
//
// The gated delta-rule recurrence is not implemented yet. For now this kernel just drains the
// K tiles the reader streams into cb_k so the producer can't deadlock. The next steps replace
// this body with the per-V-head recurrence (L2-norm/scale, decay, read, delta, write) plus the
// per-V-head tree reduction and the final o = h @ q.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_k_tiles = get_arg_val<uint32_t>(0);  // K tiles this core will consume

    constexpr uint32_t cb_k = tt::CBIndex::c_0;
    CircularBuffer cb_k_o(cb_k);

    for (uint32_t t = 0; t < num_k_tiles; ++t) {
        cb_k_o.wait_front(1);
        cb_k_o.pop_front(1);
    }
}
