// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// comm_skeleton probe 1: CIRCULAR-BUFFER BOOKKEEPING WITH NO PAYLOAD.
//
// One kernel source, two roles (PRODUCER / CONSUMER), selected by a compile-time arg so the two
// RISC-Vs of the pair compile from the same text and no role can accidentally drift from the other.
// The loop body is EXACTLY the CB cycle the real op pays around every chunk of its eltwise chains
// and weight streams (`cb_reserve_back` / `cb_push_back` on the producer side, `cb_wait_front` /
// `cb_pop_front` on the consumer side) and NOTHING ELSE — no NoC read fills the pages, no compute
// reads them. Whatever this measures is pure bookkeeping by construction.
//
// Why the loop cannot be optimised away: `cb_reserve_back`/`cb_wait_front` spin on a volatile
// `reg_read` of the CB's stream register and `cb_push_back`/`cb_pop_front` store through a
// `volatile tt_reg_ptr uint32_t*`, so every iteration is an observable side effect.
//
// PAGES_PER_CALL is the knob that separates "cost per CALL" from "cost per PAGE": at a fixed total
// page count, a per-call cost makes time scale as 1/PAGES_PER_CALL while a per-page cost leaves it
// flat. That is the `PerChunk` (blk_in/blk_out) question from eltwise_convenience.hpp, asked
// directly.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

// --- roles ---
#define ROLE_PRODUCER 0
#define ROLE_CONSUMER 1
#define ROLE_BULK_PRODUCER \
    2  // pushes the whole run in ONE call, so a paired CONSUMER's waits are
       // all pre-satisfied and its loop measures wait_front/pop_front alone.

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t PAGES_PER_CALL = get_compile_time_arg_val(1);
    constexpr uint32_t CB_ID = get_compile_time_arg_val(2);

    // RUNTIME, not compile-time: the op's own trip counts are runtime values (they fall out of the
    // token count), so a runtime bound is the faithful loop — and it lets one compiled kernel serve
    // the whole count sweep instead of one JIT build per point.
    const uint32_t N_ITERS = get_arg_val<uint32_t>(0);

    if constexpr (ROLE == ROLE_PRODUCER) {
        for (uint32_t i = 0; i < N_ITERS; ++i) {
            cb_reserve_back(CB_ID, PAGES_PER_CALL);
            cb_push_back(CB_ID, PAGES_PER_CALL);
        }
    } else if constexpr (ROLE == ROLE_CONSUMER) {
        for (uint32_t i = 0; i < N_ITERS; ++i) {
            cb_wait_front(CB_ID, PAGES_PER_CALL);
            cb_pop_front(CB_ID, PAGES_PER_CALL);
        }
    } else {  // ROLE_BULK_PRODUCER
        const uint32_t total = N_ITERS * PAGES_PER_CALL;
        cb_reserve_back(CB_ID, total);
        cb_push_back(CB_ID, total);
    }
}
