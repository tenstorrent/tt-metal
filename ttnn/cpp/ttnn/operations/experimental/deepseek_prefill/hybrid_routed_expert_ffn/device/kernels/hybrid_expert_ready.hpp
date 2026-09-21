// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The per-expert handoff from the routed-expert half to the combine half.
//
// The two halves run concurrently on disjoint rectangles, so nothing in the program orders them:
// this counter is the ONLY thing standing between combine reading an expert's rows out of DRAM
// and the routed-expert writers having put them there.
//
// WHAT IS COUNTED. Not "experts finished" -- expert INDICES WALKED. Each half loops
// local_expert_id over 0..experts_per_chip-1 and skips the experts outside its token band, and a
// skipped index is trivially done, so both bump once per index either way. That is what makes the
// count monotone in the index: after W*(e+1) increments from a half, that half is provably past
// index e. Counting only the experts a half actually ran would not be -- the fused half owns the
// low-count experts and the unified half the rest, so completion order interleaves the two index
// sequences and a bare "experts done" total says nothing about WHICH.
//
// WHY ONE COUNTER IS ENOUGH FOR TWO HALVES. The passes are separated by hybrid_pass_barrier(), so
// no core starts pass B until every core has finished pass A. Both halves therefore bump the same
// counter and the last pass's walk is the one that matters: expert e is safe at
// PRIOR + W*(e+1), where PRIOR = W * experts_per_chip * (passes - 1) accounts for the earlier
// pass having already contributed its full walk. A second counter would buy nothing -- the
// barrier already forces pass A to be globally complete before pass B's first increment.
//
// WHICH INDEX. Combine locates expert `local_expert`'s rows at `my_expert_base + local_expert` in
// the offsets table; the routed expert locates the same rows at `global_expert_idx_table[e]`. The
// gate pairs those two by position, which is sound for the same reason the two-op pipeline is:
// they already have to name the same region for the same expert, or the unoverlapped forward
// would mix experts up too. This gate inherits that invariant, it does not add one.
//
// Compiles to nothing on a kernel the program factory did not hand the defines to -- which is
// every kernel when the combine half is not carried.

#pragma once

#include <cstdint>

#if defined(HYB_EXPERT_READY_TARGETS) || defined(HYB_EXPERT_READY_WRITERS)
#include "api/dataflow/dataflow_api.h"
#endif

#ifdef HYB_EXPERT_READY_TARGETS

#ifndef HYB_EXPERT_READY_RT_BASE
#error "the routed-expert writer needs -DHYB_EXPERT_READY_RT_BASE from the program factory"
#endif

// Producer: the routed-expert writers, once per expert INDEX, and only ever after the output
// write-back for that index has been barriered. An increment that overtakes an outstanding write
// tells combine to read rows still in flight.
//
// One unicast per gating core rather than a fan-in through an aggregator, because an aggregator
// would have to spin to forward and every core here is a worker with matmuls left to run. The
// bill is TARGETS atomics per expert per writer, issued back to back and drained once.
//
// `rt_base` is where the factory appended the target list: HYB_EXPERT_READY_TARGETS pairs of NoC
// coordinates, after both halves' argument blocks and the pass barrier's.
FORCE_INLINE void hybrid_expert_ready_signal() {
    const uint32_t addr = get_semaphore(HYB_EXPERT_READY_SEM_ID);
    for (uint32_t i = 0; i < HYB_EXPERT_READY_TARGETS; ++i) {
        const uint32_t x = get_arg_val<uint32_t>(HYB_EXPERT_READY_RT_BASE + 2 * i + 0);
        const uint32_t y = get_arg_val<uint32_t>(HYB_EXPERT_READY_RT_BASE + 2 * i + 1);
        noc_semaphore_inc(get_noc_addr(x, y, addr), 1);
    }
    // The atomics must LAND, not merely depart: the next thing this kernel does is the next
    // expert, and combine may not start on this one until the count is visible.
    noc_async_atomic_barrier();
}

#else

FORCE_INLINE void hybrid_expert_ready_signal() {}

#endif

#ifdef HYB_EXPERT_READY_WRITERS

// Consumer: combine's reader and untilizer, before either touches expert `expert`'s rows. Spins
// on this core's own copy, which every routed-expert writer increments. Raw volatile with an
// explicit invalidate, matching how combine waits on its own ring semaphores.
//
// Waiting on an expert this core owns no batches of costs nothing: the count is monotone in the
// index, so a core that only wants expert N would wait exactly as long gating on N directly.
FORCE_INLINE void hybrid_expert_ready_wait(uint32_t expert) {
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(HYB_EXPERT_READY_SEM_ID));
    const uint32_t target = HYB_EXPERT_READY_PRIOR + HYB_EXPERT_READY_WRITERS * (expert + 1);
    while (*sem < target) {
        invalidate_l1_cache();
    }
}

#else

FORCE_INLINE void hybrid_expert_ready_wait(uint32_t) {}

#endif
