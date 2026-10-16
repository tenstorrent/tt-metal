// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// in1 reader for a matmul whose weights arrive over PrefetcherPipes and whose output is sharded in
// place, as the Metal 2.0 gather_in0 matmul binds it. It reads no weights itself: it drains the pipe
// into compute and waits for compute to finish the output.
//
// The binding and argument names below are this kernel's interface: every factory that later binds it
// inherits them and cannot rename them.

#include <stdint.h>

#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/matmul/device/kernels/dataflow/prefetcher_pipe_in1_window.hpp"

void kernel_main() {
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    constexpr auto num_blocks = get_arg(args::num_blocks);
    constexpr auto out_block_num_tiles = get_arg(args::out_block_num_tiles);

    const Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    // The producer streams this worker's K-blocks in the order compute consumes them -- for
    // gather_in0, ring-rotated with the worker's own K-block first, matching the in0 shard compute
    // starts on. One accessor names every pipe; the one present on this worker is the one bound here.
    // bind_relay() aligns in1 to the pipe's durable cursor (firmware resets it at launch) and makes
    // pop_front wait for compute. The pipe lives to the end of kernel_main; its destructor stores the
    // cursor back.
    experimental::PrefetcherPipe pipe(pipe::in1);
    auto in1_relay = pipe.bind_relay();
    for (uint32_t block = 0; block < num_blocks; ++block) {
        publish_in1_block_from_pipe(pipe, in1_relay, noc, block, in1_block_num_tiles);
    }
    drain_in1_pipe_window(pipe, noc, num_blocks);

    // The output stays resident in the sharded output tensor; wait for compute to finish it so this
    // buffer has the consumer every buffer needs.
    dfb_out.wait_front(out_block_num_tiles);

    // pop_front acks the sender with non-posted NOC atomics; retire them before the kernel exits.
    noc.async_atomic_barrier();
}
