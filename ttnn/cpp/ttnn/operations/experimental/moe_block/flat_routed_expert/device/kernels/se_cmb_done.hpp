// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Combine overlap: every y writer (down core) reports its progress to the combine's collector as its rows land, so
// combine takes rows while the flat expert still computes the next ones. Step s is local expert s (slot order); the
// collector counts per step (cumulative counts) and releases step s once every writer has reported it.
//
// A writer reports the longest finished prefix of steps: it counts the blocks it still owes per local expert; the
// caller guarantees the rows have landed (`barrier`: or this waits for its writes).
//
// Runtime args at SE_CMB_DONE_RT: collector noc x, collector noc y, the collector's count array address, (unused).
#pragma once

#ifdef SE_CMB_DONE_RT
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"

struct SeCmbDone {
    uint16_t left[SE_MAX_E];  // blocks this core still writes, per local expert (step = local expert)
    uint32_t num_steps = 0, next = 0;

    // `writes`: this core writes y blocks for the schedule's entries (false: it reports every step at once)
    void init(const SeDyn& d, uint32_t ne, bool writes) {
        num_steps = ne;
        for (uint32_t e = 0; e < ne; ++e) {
            left[e] = 0;
        }
        for (uint32_t a = 0; writes && a < d.n_act; ++a) {
            left[d.eid[a]] += d.subs[a];
        }
    }
    // One block of local expert e written.
    void wrote(uint32_t e) { --left[e]; }
    bool step_done(uint32_t s) const { return left[s] == 0; }
    // Report every finished step not reported yet (all earlier ones too). `barrier`: wait for this core's writes to
    // land first (false: the caller knows they have, the row-major y writer's flushed transaction ids).
    void report(bool barrier = true) {
        if (next >= num_steps || !step_done(next)) {
            return;
        }
        if (barrier) {
            noc_async_write_barrier();  // the rows have landed
        }
        const uint32_t x = get_arg_val<uint32_t>(SE_CMB_DONE_RT);
        const uint32_t y = get_arg_val<uint32_t>(SE_CMB_DONE_RT + 1);
        const uint32_t counts = get_arg_val<uint32_t>(SE_CMB_DONE_RT + 2);
        while (next < num_steps && step_done(next)) {
            noc_semaphore_inc(get_noc_addr(x, y, counts + 4 * next), 1);
            ++next;
        }
        noc_async_atomic_barrier();
    }
};
#endif
