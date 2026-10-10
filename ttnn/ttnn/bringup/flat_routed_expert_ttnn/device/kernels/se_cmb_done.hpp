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
    // Report every step whose expert this core has finished (all earlier ones too). `barrier`: wait for this core's
    // writes to land first (false: the caller knows they have, the row-major y writer's flushed transaction ids).
    void report(bool barrier = true) {
        if (next >= num_e || left[order[next]] != 0) {
            return;
        }
        if (!go_seen) {
            volatile tt_l1_ptr uint32_t* go =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(SE_CMB_DONE_RT + 3));
            while (*go == 0) {
                invalidate_l1_cache();
            }
            *go = 0;  // the collector sets it once per launch
            go_seen = true;
        }
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
