// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Combine overlap (flat_combine_overlap): every y writer (down core, reader tail) reports each local expert to
// combine_fabric2d's collector once its rows of that expert have landed, so combine takes the expert while the flat
// expert still computes the next ones. The collector counts per step and releases step s once every writer has
// reported it. Step s is the s-th expert in the order this chip's schedule finishes them (se_dyn_completion_order:
// inactive experts first, then by each expert's last entry, so a pinned expert comes at its last chunk); combine
// rebuilds the same order from the counts and walks the experts in it.
//
// A writer reports step s once it has written all of its blocks of the experts of steps 0..s: it counts the blocks
// it still owes per expert and reports the longest finished prefix of the order, behind a write barrier (the rows
// have landed, not just left). The collector
// zeroes its counts at launch and then multicasts `go`; a writer waits for `go` before its first report.
//
// Runtime args at SE_CMB_DONE_RT: collector noc x, collector noc y, the collector's count array address, go address.
#pragma once

#ifdef SE_CMB_DONE_RT
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"

struct SeCmbDone {
    uint16_t left[SE_MAX_E];  // blocks this core still writes, per local expert
    uint8_t order[SE_MAX_E];  // step -> local expert (se_dyn_completion_order)
    uint32_t num_e = 0, next = 0;
    bool go_seen = false;
#ifdef SE_CMB_PROGRESSIVE
    // Progressive release: an expert the schedule splits into several entries (pinning) is also reported entry by
    // entry, as (step, chunk) = (its step, the entry's rank among its entries) on the collector's chunk counts (the
    // layout of hybrid_expert_done.hpp: area at counts + 1024, 16 chunks per step, 64 steps), so combine can read
    // the entry's rows before the whole expert is done.
    static constexpr uint32_t CHUNK_AREA = 1024, MAX_CHUNKS = 16, MAX_STEPS = 64;
    uint16_t ent_left[SE_MAX_V];  // sub-blocks of entry a this core still writes
    uint8_t ent_chunk[SE_MAX_V];  // entry a's rank among its expert's entries
    uint8_t n_chunks[SE_MAX_E];   // entries per local expert
    uint8_t step_of[SE_MAX_E];    // local expert -> step
#endif

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
        num_e = ne;
#ifdef SE_CMB_SLOT_ORDER  // combine walks slot order (bfp8 tiles through its untilizers, or the probe)
        for (uint32_t e = 0; e < ne; ++e) {
            order[e] = static_cast<uint8_t>(e);
        }
#else
        se_dyn_completion_order(d, ne, order);
#endif
        for (uint32_t e = 0; e < ne; ++e) {
            left[e] = 0;
        }
        if (writes) {
            for (uint32_t a = 0; a < d.n_act; ++a) {
                left[d.eid[a]] += d.subs[a];
            }
        }
#ifdef SE_CMB_PROGRESSIVE
        for (uint32_t e = 0; e < ne; ++e) {
            n_chunks[e] = 0;
        }
        for (uint32_t s = 0; s < ne; ++s) {
            step_of[order[s]] = static_cast<uint8_t>(s);
        }
        for (uint32_t a = 0; a < d.n_act; ++a) {
            ent_left[a] = writes ? d.subs[a] : 0;
            ent_chunk[a] = n_chunks[d.eid[a]]++;
        }
#endif
#ifdef SE_CMB_EARLY  // (perf probe: every step reported at launch, combine runs ungated beside the flat expert)
        for (uint32_t e = 0; e < ne; ++e) {
            left[e] = 0;
        }
        report();
#endif
    }
    void wrote(uint32_t e) {
#ifndef SE_CMB_EARLY
        --left[e];
#endif
    }
    // One sub-block of schedule entry a written (and landed: the row-major writer's flushed transaction ids). The
    // entry's last one reports its chunk if its expert has several entries (no-op without SE_CMB_PROGRESSIVE).
    void wrote_entry(const SeDyn& d, uint32_t a) {
#if defined(SE_CMB_PROGRESSIVE) && !defined(SE_CMB_EARLY)
        if (--ent_left[a] != 0) {
            return;
        }
        const uint32_t e = d.eid[a], step = step_of[e], chunk = ent_chunk[a];
        if (n_chunks[e] < 2 || chunk >= MAX_CHUNKS || step >= MAX_STEPS) {
            return;  // the whole-expert report covers it
        }
        wait_go();
        const uint32_t x = get_arg_val<uint32_t>(SE_CMB_DONE_RT);
        const uint32_t y = get_arg_val<uint32_t>(SE_CMB_DONE_RT + 1);
        const uint32_t area = get_arg_val<uint32_t>(SE_CMB_DONE_RT + 2) + CHUNK_AREA;
        noc_semaphore_inc(get_noc_addr(x, y, area + 4 * (step * MAX_CHUNKS + chunk)), 1);
        noc_async_atomic_barrier();
#else
        (void)d, (void)a;
#endif
    }
    // Report every step whose expert this core has finished (all earlier ones too). `barrier`: wait for this core's
    // writes to land first (false: the caller knows they have, the row-major y writer's flushed transaction ids).
    void report(bool barrier = true) {
        if (next >= num_e || left[order[next]] != 0) {
            return;
        }
        wait_go();
        if (barrier) {
            noc_async_write_barrier();  // the rows have landed
        }
        const uint32_t x = get_arg_val<uint32_t>(SE_CMB_DONE_RT);
        const uint32_t y = get_arg_val<uint32_t>(SE_CMB_DONE_RT + 1);
        const uint32_t counts = get_arg_val<uint32_t>(SE_CMB_DONE_RT + 2);
        while (next < num_e && left[order[next]] == 0) {
            noc_semaphore_inc(get_noc_addr(x, y, counts + 4 * next), 1);
            ++next;
        }
        noc_async_atomic_barrier();
    }
};
#endif
