// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) reader for binary_ng's no-broadcast ADD, fully-sharded case.
//
// For a fully-sharded input the source tiles are already L1-resident (the DFB borrows the input
// tensor's shard via DataflowBufferSpec::borrowed_from). There is no NoC read to perform: the
// reader simply publishes the resident shard's tiles into the DFB so the compute kernel can consume
// them. This is the DFB analog of the CB reader's `#if SRC_SHARDED` short-circuit in
// reader_interleaved_no_bcast.cpp (bulk reserve_back + push_back, no NoC traffic).
//
// Two input DFBs (in0, in1) are published. `num_tiles` is this core's shard tile count.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"  // get_my_thread_id / get_num_threads (multi-NEO partition)
#include "experimental/kernel_args.h"

#ifdef ENABLE_KERNEL_TIMER
#include "api/debug/kernel_timer.h"
constexpr uint32_t kTimerSlotReader = 0;  // reader=0, compute=1, writer=2
#endif

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

#ifdef ENABLE_KERNEL_TIMER
    KernelTimer _timer;
    _timer.start();
#endif

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    const uint32_t nthreads = get_num_threads();
    if (nthreads <= 1) {
        // Single-NEO fast path: bulk publish the whole resident shard (original behavior).
        dfb_in0.reserve_back(num_tiles);
        dfb_in0.push_back(num_tiles);
        dfb_in1.reserve_back(num_tiles);
        dfb_in1.push_back(num_tiles);
    } else {
        // Multi-NEO: N reader threads (one per NEO) feed a STRIDED DFB. Publish ONETILE at a time with
        // modulo-skip so each thread's credits land on its own tile-counter, matching the compute
        // consumer's strided onetile consumption (bulk count-division mis-aligns the round-robin).
        const uint32_t tid = get_my_thread_id();
        for (uint32_t t = 0; t < num_tiles; ++t) {
            if (t % nthreads != tid) {
                continue;
            }
            dfb_in0.reserve_back(1);
            dfb_in0.push_back(1);
            dfb_in1.reserve_back(1);
            dfb_in1.push_back(1);
        }
    }

#ifdef ENABLE_KERNEL_TIMER
    kernel_timer_write(get_arg(args::timer_l1_addr), kTimerSlotReader, _timer.stop());
#endif
}
