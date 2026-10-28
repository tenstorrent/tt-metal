// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The in1 side of a matmul whose weights arrive over a PrefetcherPipe, shared by every in1 reader that
// drains one. One pipe entry is one K-block, and in1 is a relay over the pipe's ring that pages each
// entry as its tiles, so the K-blocks are consumed in place. The reader keeps a window of
// kIn1PipeWindowBlocks K-blocks unacked: it publishes the newest to compute while compute drains the
// older ones, and hands an entry back to the sender only once compute has popped its tiles.

#pragma once

#include <stdint.h>

#ifdef ARCH_QUASAR
#error "PrefetcherPipe weight delivery into the matmul has not been brought up on Quasar"
#endif

#include "api/dataflow/noc.h"
#include "api/dataflow/prefetcher_pipe.h"

// K-blocks the reader holds unacked: the one it publishes plus one compute may still be draining. The
// host sizes the ring to hold this many (kLookaheadMinResidentBlocks in matmul_device_operation.cpp).
constexpr uint32_t kIn1PipeWindowBlocks = 2;

// Waits for K-block `block` to land and publishes it to compute, then hands back the oldest entry once
// the window is full. `block` counts from 0 over one pass of K-blocks, which drain_in1_pipe_window
// ends. wait_front counts from the oldest unacked entry. Publish only through the relay: pushing in1
// as well would double the credit compute sees.
FORCE_INLINE void publish_in1_block_from_pipe(
    experimental::PrefetcherPipe& pipe,
    experimental::PrefetcherPipe::RelayView& in1_relay,
    const Noc& noc,
    uint32_t block,
    uint32_t in1_block_num_tiles) {
    in1_relay.reserve_back(in1_block_num_tiles);
    pipe.wait_front(block + 1 < kIn1PipeWindowBlocks ? block + 1 : kIn1PipeWindowBlocks);
    in1_relay.push_back(in1_block_num_tiles);
    if (block + 1 >= kIn1PipeWindowBlocks) {
        // Waits for compute to pop that block's tiles before acking the sender.
        pipe.pop_front(1, noc);
    }
}

// After the last of `num_blocks` K-blocks: hands back the entries the window still holds, each once
// compute has drained it.
FORCE_INLINE void drain_in1_pipe_window(experimental::PrefetcherPipe& pipe, const Noc& noc, uint32_t num_blocks) {
    const uint32_t held = num_blocks < kIn1PipeWindowBlocks - 1 ? num_blocks : kIn1PipeWindowBlocks - 1;
    if (held > 0) {
        pipe.pop_front(held, noc);
    }
}
