// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) writer for binary_ng's no-broadcast ADD, fully-sharded output.
//
// For a fully-sharded output the compute kernel packs directly into the output DFB, which borrows
// the output tensor's shard (DataflowBufferSpec::borrowed_from). There is no NoC write to perform:
// the writer just drains the output DFB so its credits return to the compute producer. This is the
// DFB analog of the CB writer's `#if DST_SHARDED` skip (writer_interleaved_no_bcast.cpp — all
// noc.async_write calls are compiled out, leaving only FIFO sync).
//
// `num_tiles` is this core's shard tile count.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"  // get_my_thread_id / get_num_threads (multi-NEO partition)
#include "experimental/kernel_args.h"

#ifdef ENABLE_KERNEL_TIMER
#include "api/debug/kernel_timer.h"
constexpr uint32_t kTimerSlotWriter = 2;  // reader=0, compute=1, writer=2
#endif

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

#ifdef ENABLE_KERNEL_TIMER
    KernelTimer _timer;
    _timer.start();
#endif

    DataflowBuffer dfb_out(dfb::out);

    const uint32_t nthreads = get_num_threads();
    if (nthreads <= 1) {
        // Single-NEO fast path: bulk drain (original behavior).
        dfb_out.wait_front(num_tiles);
        dfb_out.pop_front(num_tiles);
    } else {
        // Multi-NEO: N writer threads (one per NEO) drain a STRIDED output DFB. Drain ONETILE at a
        // time with modulo-skip so each thread consumes the credits produced by the matching compute
        // thread on its own tile-counter (bulk count-division mis-aligns the round-robin and deadlocks
        // or corrupts).
        const uint32_t tid = get_my_thread_id();
        for (uint32_t t = 0; t < num_tiles; ++t) {
            if (t % nthreads != tid) {
                continue;
            }
            dfb_out.wait_front(1);
            dfb_out.pop_front(1);
        }
    }

#ifdef ENABLE_KERNEL_TIMER
    kernel_timer_write(get_arg(args::timer_l1_addr), kTimerSlotWriter, _timer.stop());
#endif
}
