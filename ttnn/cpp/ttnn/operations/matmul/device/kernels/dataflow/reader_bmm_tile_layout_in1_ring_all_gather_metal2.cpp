// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in1_ring_all_gather.cpp, which lives beside it. The Metal
// 2.0 gather_in0 matmul binds this fork; the original serves the legacy MeshWorkload builder and the
// fused reduce-scatter matmul.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// This fork carries one in1 transport, PrefetcherPipe delivery, the only one the Metal 2.0 gather_in0
// factory accepts. The legacy kernel's DRAM, L1-sharded and GlobalCircularBuffer paths are not
// carried.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef ARCH_QUASAR
#error "PrefetcherPipe weight delivery into this matmul has not been brought up on Quasar"
#endif
#include "api/dataflow/prefetcher_pipe.h"

void kernel_main() {
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    constexpr auto num_blocks = get_arg(args::num_blocks);
    constexpr auto out_block_num_tiles = get_arg(args::out_block_num_tiles);

    const Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    // in1 is a relay laid over this worker's PrefetcherPipe ring, so the K-blocks arrive already in
    // place: this kernel only turns a delivered entry (one K-block) into in1 credit for compute (its
    // tiles, one relay page each) and, once compute is done with it, that entry's credit back into
    // an ack to the sender. The producer streams this worker's K-blocks in ring-rotated order --
    // its own K-block first, matching the in0 shard compute starts on -- so they are consumed in
    // arrival order. One accessor names every pipe; the one present on this worker is the one bound
    // here. bind_relay() aligns in1 to the pipe's durable cursor (firmware resets it at launch) and
    // makes pop_front wait for compute. The pipe lives to the end of kernel_main; its destructor
    // stores the cursor back.
    experimental::PrefetcherPipe pipe(pipe::in1);
    auto in1_relay = pipe.bind_relay();

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // One K-block of lookahead: publish this block to compute, then hand the previous block's
        // entry back to the sender once compute has drained it. wait_front counts from the oldest
        // unacked entry, which trails this block by one after the first.
        in1_relay.reserve_back(in1_block_num_tiles);
        pipe.wait_front(block == 0 ? 1u : 2u);
        // Publish only through the relay view: pushing in1 as well would double the credit compute
        // sees. pop_front waits for compute to have popped that block's tiles before acking it.
        in1_relay.push_back(in1_block_num_tiles);
        if (block >= 1) {
            pipe.pop_front(1, noc);
        }
    }
    if constexpr (num_blocks > 0) {
        pipe.pop_front(1, noc);
    }

    // The output stays resident in the sharded output tensor; wait for compute to finish it so this
    // buffer has the consumer every buffer needs.
    dfb_out.wait_front(out_block_num_tiles);

    // pop_front acks the sender with non-posted NOC atomics; retire them before the kernel exits.
    noc.async_atomic_barrier();
    noc.async_write_barrier();
}
