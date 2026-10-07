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
    // Whether the producer sends this worker's K-blocks in ring order, its own first, or in K order.
    constexpr bool in1_in_ring_order = get_arg(args::in1_in_ring_order);
    // This worker's position in the gather ring.
    const uint32_t ring_idx = get_arg(args::ring_idx);

    const Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    // Compute consumes the K-blocks in ring order, its own first. One accessor names every pipe; the
    // one present on this worker is the one bound here. bind_relay() aligns in1 to the pipe's durable
    // cursor (firmware resets it at launch) and makes pop_front wait for compute. The pipe lives to
    // the end of kernel_main; its destructor stores the cursor back.
    experimental::PrefetcherPipe pipe(pipe::in1);
    auto in1_relay = pipe.bind_relay();
    if constexpr (in1_in_ring_order) {
        // They arrive in that order, so each is handed back as soon as compute has drained it.
        for (uint32_t block = 0; block < num_blocks; ++block) {
            publish_in1_block_from_pipe(pipe, in1_relay, noc, block, in1_block_num_tiles);
        }
        drain_in1_pipe_window(pipe, noc, num_blocks);
    } else {
        // They arrive in K order, so the whole layer stays in the ring until compute is done with it.
        // Compute walks it in ring order by stepping over what it does not need yet: publish the layer
        // as it lands, and then, unless this worker's own K-block is the layer's first, the rest of
        // the ring, which holds none of the layer, and the layer's first ring_idx K-blocks again, which
        // compute reaches by coming back round.
        for (uint32_t block = 0; block < num_blocks; ++block) {
            in1_relay.reserve_back(in1_block_num_tiles);
            pipe.wait_front(block + 1);
            in1_relay.push_back(in1_block_num_tiles);
        }
        if (ring_idx > 0) {
            // bind_relay() sized in1 to the pipe's ring.
            const uint32_t rest_of_ring_tiles =
                DataflowBuffer(dfb::in1).get_total_num_entries() - num_blocks * in1_block_num_tiles;
            if (rest_of_ring_tiles > 0) {
                in1_relay.reserve_back(rest_of_ring_tiles);
                in1_relay.push_back(rest_of_ring_tiles);
            }
            for (uint32_t block = 0; block < ring_idx; ++block) {
                in1_relay.reserve_back(in1_block_num_tiles);
                in1_relay.push_back(in1_block_num_tiles);
            }
        }
    }

    // The output stays resident in the sharded output tensor; wait for compute to finish it so this
    // buffer has the consumer every buffer needs.
    dfb_out.wait_front(out_block_num_tiles);

    if constexpr (!in1_in_ring_order) {
        // Compute has read the whole layer: hand it back to the sender.
        pipe.pop_front(num_blocks, noc);
    }

    // pop_front acks the sender with non-posted NOC atomics; retire them before the kernel exits.
    noc.async_atomic_barrier();
}
