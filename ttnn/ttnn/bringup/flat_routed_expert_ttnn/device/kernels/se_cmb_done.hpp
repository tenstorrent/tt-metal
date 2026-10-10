// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Combine overlap (flat_combine_overlap): every y writer (down core, reader tail) reports its progress to
// combine_fabric2d's collector as its rows land, so combine takes rows while the flat expert still computes the next
// ones. The collector counts per step and releases step s once every writer has reported it.
//
// Steps are the schedule's entries (default, SE_CMB_STEPS = the collector's step count): step s is entry s of this
// chip's schedule (se_dyn.hpp: one active expert, or one chunk of a pinned expert, as a row range of its region);
// entries finish in schedule order, so a pinned (hot) expert is released chunk by chunk as it is computed. Steps past
// the last entry are empty and reported at once. Combine rebuilds every chip's schedule from the counts and walks
// the same entries. With SE_CMB_SLOT_ORDER (bfp8 tiles through combine's untilizers, or the probe) step s is local
// expert s instead.
//
// A writer reports the longest finished prefix of steps: it counts the sub-blocks (entries) or blocks (slot order)
// it still owes; the caller guarantees the rows have landed (`barrier`: or this waits for its writes). The collector
// zeroes its counts at launch and then multicasts `go`; a writer waits for `go` before its first report.
//
// Runtime args at SE_CMB_DONE_RT: collector noc x, collector noc y, the collector's count array address, go address.
#pragma once

#ifdef SE_CMB_DONE_RT
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"

struct SeCmbDone {
#ifdef SE_CMB_SLOT_ORDER
    uint16_t left[SE_MAX_E];  // blocks this core still writes, per local expert (step = local expert)
#else
    uint16_t left[SE_MAX_V];  // sub-blocks this core still writes, per schedule entry (step = entry)
    uint32_t n_ent = 0;
#endif
    uint32_t num_steps = 0, next = 0;
    bool go_seen = false;

    void wait_go() {
        if (go_seen) {
            return;
        }
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(SE_CMB_DONE_RT + 3));
        while (*go == 0) {
            invalidate_l1_cache();
        }
        *go = 0;  // the collector sets it once per launch
        go_seen = true;
    }

    // `writes`: this core writes y blocks for the schedule's entries (false: it reports every step at once)
    void init(const SeDyn& d, uint32_t ne, bool writes) {
#ifdef SE_CMB_SLOT_ORDER
        num_steps = ne;
        for (uint32_t e = 0; e < ne; ++e) {
            left[e] = 0;
        }
        for (uint32_t a = 0; writes && a < d.n_act; ++a) {
            left[d.eid[a]] += d.subs[a];
        }
#else
        (void)ne;
        num_steps = SE_CMB_STEPS;
        n_ent = d.n_act;
        for (uint32_t a = 0; a < d.n_act; ++a) {
            left[a] = writes ? d.subs[a] : 0;
        }
#endif
#ifdef SE_CMB_EARLY  // (perf probe: every step reported at launch, combine runs ungated beside the flat expert)
#ifdef SE_CMB_SLOT_ORDER
        for (uint32_t e = 0; e < ne; ++e) {
            left[e] = 0;
        }
#else
        n_ent = 0;
#endif
        report();
#endif
    }
    // One block of local expert e / one sub-block of schedule entry a written (each mode uses its own).
    void wrote(uint32_t e) {
#if defined(SE_CMB_SLOT_ORDER) && !defined(SE_CMB_EARLY)
        --left[e];
#else
        (void)e;
#endif
    }
    void wrote_entry(const SeDyn& d, uint32_t a) {
        (void)d;
#if !defined(SE_CMB_SLOT_ORDER) && !defined(SE_CMB_EARLY)
        --left[a];
#else
        (void)a;
#endif
    }
    bool step_done(uint32_t s) const {
#ifdef SE_CMB_SLOT_ORDER
        return left[s] == 0;
#else
        return s >= n_ent || left[s] == 0;
#endif
    }
    // Report every finished step not reported yet (all earlier ones too). `barrier`: wait for this core's writes to
    // land first (false: the caller knows they have, the row-major y writer's flushed transaction ids).
    void report(bool barrier = true) {
        if (next >= num_steps || !step_done(next)) {
            return;
        }
        wait_go();
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
