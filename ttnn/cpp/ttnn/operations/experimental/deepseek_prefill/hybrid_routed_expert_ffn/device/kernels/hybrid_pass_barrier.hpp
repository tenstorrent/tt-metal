// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The grid-wide rendezvous between the union kernel's two passes.
//
// Both halves run on ALL the op's cores, one after the other, and they share L1: their circular
// buffers are laid over one arena and their semaphores are drawn from one id block. So the pass
// boundary is not per-core ordering -- core X's the second pass multicasts weights into core Y's L1, over
// the very buffers Y may still be reading for the first pass. Nothing may cross until the first pass is finished
// EVERYWHERE.
//
// Both data-movement kernels arrive, and the master waits for two arrivals per core. That is what
// makes the barrier sufficient without a separate local handshake: the writer is the last stage of
// a core's pipeline, so its the first pass completing implies that core's compute has already pushed
// everything and finished reading. Waiting on readers alone would let a core whose reader ran dry
// early release the grid while its own compute was still mid-matmul.
//
// The compute kernels do not take part, and cannot -- a TRISC has no NoC. They are gated
// implicitly: compute's first second-pass action that touches shared L1 is packing an output, which
// cannot happen until it has inputs, and those only arrive from a reader already past the barrier.
//
// The master also ZEROES the shared semaphore block before releasing. The first pass leaves those ids at
// arbitrary values and the second pass's waits assume they start at zero; doing it from the master, before
// the release, means no core can observe a half-reset block.

#pragma once

#include <cstdint>

#if defined(HYB_RUN_FUSED_PASS) && !defined(COMPILE_FOR_TRISC)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

#ifndef HYB_BARRIER_RT_BASE
#error "the union kernel needs -DHYB_BARRIER_RT_BASE from the program factory"
#endif

namespace hybrid_pass_barrier_detail {

// Runtime-arg layout appended after both halves' blocks; see hybrid_half_merge.cpp.
enum Arg : uint32_t {
    IS_MASTER = 0,
    MASTER_NOC_X,
    MASTER_NOC_Y,
    RECT_X_START,
    RECT_Y_START,
    RECT_X_END,
    RECT_Y_END,
    NUM_RECEIVERS,
    TOTAL_ARRIVALS,
    BARRIER_SEM_ID,
    SHARED_SEM_COUNT,
    COUNT,
};

inline uint32_t arg(uint32_t which) { return get_arg_val<uint32_t>(HYB_BARRIER_RT_BASE + which); }

// Above any arrival count, so a core cannot mistake the tail of the arrival phase for the release.
inline constexpr uint32_t kReleased = 0xB417u;

}  // namespace hybrid_pass_barrier_detail

inline void hybrid_pass_barrier() {
    namespace d = hybrid_pass_barrier_detail;

    Noc noc;
    Semaphore<> barrier_sem(d::arg(d::BARRIER_SEM_ID));

    // Everything this kernel issued for the first pass must have landed before it claims to be done: an
    // increment that overtakes an outstanding transaction would release the grid over data still
    // moving. READS matter as much as writes here -- the two halves' circular buffers alias the
    // same arena bytes, so a pass-A read landing after the release is overwritten by the second pass.
    noc.async_write_barrier();
    noc.async_read_barrier();
    noc_async_atomic_barrier();

    barrier_sem.up(noc, d::arg(d::MASTER_NOC_X), d::arg(d::MASTER_NOC_Y), 1);

    if (d::arg(d::IS_MASTER) != 0) {
        barrier_sem.wait_min(d::arg(d::TOTAL_ARRIVALS));

        // Hand the second pass a zeroed block. Done per id through the public set/multicast pair rather
        // than as one write over the semaphore region, so the barrier's own id -- which must
        // survive -- cannot be caught by a range that grows later.
        const uint32_t shared = d::arg(d::SHARED_SEM_COUNT);
        const uint32_t receivers = d::arg(d::NUM_RECEIVERS);
        for (uint32_t id = 0; id < shared; ++id) {
            Semaphore<> shared_sem(id);
            shared_sem.set(0);
            if (receivers > 0) {
                shared_sem.set_multicast<NocOptions::DEFAULT>(
                    noc,
                    d::arg(d::RECT_X_START),
                    d::arg(d::RECT_Y_START),
                    d::arg(d::RECT_X_END),
                    d::arg(d::RECT_Y_END),
                    receivers);
            }
        }
        // The zeroing must have LANDED before anyone is let go -- a barrier, not a flush.
        // async_writes_flushed only waits for departure, and the release below is a local store
        // that frees this core's writer immediately; that writer issues the second pass on the OTHER NoC,
        // which has no ordering against this one. Its increment could then reach a remote
        // semaphore ahead of this still-in-flight zeroing multicast and be erased, hanging the
        // receiver.
        noc.async_write_barrier();

        barrier_sem.set(d::kReleased);
        if (receivers > 0) {
            barrier_sem.set_multicast<NocOptions::DEFAULT>(
                noc,
                d::arg(d::RECT_X_START),
                d::arg(d::RECT_Y_START),
                d::arg(d::RECT_X_END),
                d::arg(d::RECT_Y_END),
                receivers);
        }
    }

    barrier_sem.wait_min(d::kReleased);
}

#else

// The first pass is not run, or this is a compute kernel with no NoC of its own.
inline void hybrid_pass_barrier() {}

#endif
