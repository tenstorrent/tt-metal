// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The routed expert's side of the combine handoff: every writer reports once per expert INDEX it walks, in
// both passes, whether or not that pass had work for the expert. Report s bumps word s of a per-step count
// array on the combine collector, so the collector can tell that EVERY writer has passed step s. One shared
// counter cannot: writers skew, and one that runs ahead through slots its pass skips would make up the
// total for a writer still writing an earlier expert.
//
// The array is plain L1 the collector zeroes at launch, so no writer may bump it before the collector says
// `go`. That wait is paid once, at the first report. `go` is a global semaphore -- the routed expert's own
// program semaphores are all taken -- so it keeps its value across launches, and each writer clears its copy
// as soon as it has seen it: the collector sets it once per launch, so nothing can arrive after the clear.
//
// The writer is the last stage of a core's pipeline, so its writes landing is the expert's output being
// in DRAM. The fused half defers an M-block's output barrier to the next block, so the barrier here is
// not redundant: without it the bump can overtake the expert's last rows.
//
// Enabled by the program factory defining HYB_EXPERT_DONE_RT_BASE, the index of a four-word runtime-arg
// block [collector noc_x, collector noc_y, count array address, go address] appended after every other
// argument. Without it the hook compiles to nothing and the op runs as it does alone.

#pragma once

#if defined(HYB_EXPERT_DONE_RT_BASE) && !defined(COMPILE_FOR_TRISC)

#include "api/dataflow/dataflow_api.h"

inline uint32_t hyb_expert_done_step = 0;

inline void hyb_expert_done() {
    noc_async_write_barrier();
    const uint32_t noc_x = get_arg_val<uint32_t>(HYB_EXPERT_DONE_RT_BASE + 0);
    const uint32_t noc_y = get_arg_val<uint32_t>(HYB_EXPERT_DONE_RT_BASE + 1);
    const uint32_t counts_addr = get_arg_val<uint32_t>(HYB_EXPERT_DONE_RT_BASE + 2);
    if (hyb_expert_done_step == 0) {
        volatile tt_l1_ptr uint32_t* go =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(HYB_EXPERT_DONE_RT_BASE + 3));
        invalidate_l1_cache();
        while (*go == 0) {
            invalidate_l1_cache();
        }
        *go = 0;
    }
    noc_semaphore_inc(
        get_noc_addr(noc_x, noc_y, counts_addr + hyb_expert_done_step * static_cast<uint32_t>(sizeof(uint32_t))), 1);
    hyb_expert_done_step++;
    // Retired per expert so the kernel never exits with a NoC atomic outstanding.
    noc_async_atomic_barrier();
}

#define HYB_EXPERT_DONE() hyb_expert_done()

#else

#define HYB_EXPERT_DONE() \
    do {                  \
    } while (0)

#endif
