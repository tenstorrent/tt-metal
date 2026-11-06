// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Input pipelining shared by the minimal_matmul (Metal 2.0) and all_gather_minimal_matmul_async data-movement
// kernels, whose in0 / in1 blocks travel down the same store-and-forward chain of unicast hops.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

/**
 * Input pipelining modes, chosen by the host (minimal_matmul_input_pipelining.hpp):
 *
 * MM_CUT_THROUGH_CHUNKS = n > 0: cut-through forwarding along the store-and-forward chain, see
 *   move_block_cut_through.
 * MM_INJECTOR_PREFETCH: an injector issues the DRAM reads of its next K block into the next DFB slot, when that slot
 *   is already free, before it forwards the current block, so the read latency overlaps the forward instead of
 *   following it. Within an output block only.
 *
 * The two are exclusive: cut-through already overlaps an injector's reads with its forwarding, chunk by chunk.
 */
#ifndef MM_CUT_THROUGH_CHUNKS
#define MM_CUT_THROUGH_CHUNKS 0
#endif
#if MM_CUT_THROUGH_CHUNKS > 0 && defined(MM_INJECTOR_PREFETCH)
#error "MM_CUT_THROUGH_CHUNKS and MM_INJECTOR_PREFETCH are exclusive"
#endif
// The fused-CCL k-block schedule (FUSE_AG) is only compiled by the legacy kernels, which have neither mode.
#if defined(FUSE_AG) && (MM_CUT_THROUGH_CHUNKS > 0 || defined(MM_INJECTOR_PREFETCH))
#error "MM_CUT_THROUGH_CHUNKS / MM_INJECTOR_PREFETCH do not support FUSE_AG"
#endif

/**
 * Move one input block into this core's reserved DFB slot and on down the forwarding chain, cut-through.
 *
 * The store-and-forward path lands a whole block on a core before that core forwards it, so a block reaches the
 * far end of an L-core chain only after L whole-block transfers, and an injector reads a block from DRAM and only
 * then forwards it. Here the block moves in `num_chunks` row chunks: a core forwards chunk c as soon as it has
 * landed (and its downstream core has signalled a free slot), and an injector reads chunk c + 1 from DRAM while it
 * forwards chunk c. Compute still sees whole blocks: push_block() runs once every chunk has landed.
 *
 * Protocol: a receiver clears its receiver semaphore and signals the upstream core's sender semaphore, as in the
 * store-and-forward path; the upstream core then increments the receiver semaphore once per landed chunk instead of
 * relaying VALID once, so `receiver_sem >= c + 1` means chunks 0..c are in L1. Chunks are written to the same slot
 * offset downstream, which holds because every core of a chain advances its DFB ring identically.
 *
 * read_chunk(c):    injector only - issue the reads of chunk c into the slot, without a barrier.
 * forward_chunk(c): issue the writes of chunk c to the downstream core's slot, without a flush.
 * push_block():     push the whole block to compute.
 */
template <bool is_injector, typename SemT, typename ReadChunk, typename ForwardChunk, typename PushBlock>
inline void move_block_cut_through(
    const Noc& noc,
    SemT& sender_sem,
    SemT& receiver_sem,
    bool forward,
    uint32_t upstream_noc_x,
    uint32_t upstream_noc_y,
    uint32_t downstream_noc_x,
    uint32_t downstream_noc_y,
    uint32_t num_chunks,
    ReadChunk&& read_chunk,
    ForwardChunk&& forward_chunk,
    PushBlock&& push_block) {
    bool downstream_ready = false;
    uint32_t forwarded = 0;
    // Forward every landed chunk not yet forwarded, if the downstream slot is free; never blocks on it.
    auto forward_landed = [&](uint32_t landed) {
        if (!downstream_ready) {
            if (sender_sem.value() != 1) {
                return;
            }
            sender_sem.set(0);
            downstream_ready = true;
        }
        for (; forwarded < landed; forwarded++) {
            forward_chunk(forwarded);
            noc.async_writes_flushed();
            receiver_sem.up(noc, downstream_noc_x, downstream_noc_y, 1);
        }
    };

    if constexpr (is_injector) {
        read_chunk(0);
        for (uint32_t c = 0; c < num_chunks; c++) {
            noc.async_read_barrier();
            noc.write_zeros_l1_barrier();
            if (c + 1 < num_chunks) {
                read_chunk(c + 1);
            } else {
                push_block();
            }
            if (forward) {
                forward_landed(c + 1);
            }
        }
    } else {
        receiver_sem.set(INVALID);
        sender_sem.up(noc, upstream_noc_x, upstream_noc_y, 1);
        for (uint32_t c = 0; c < num_chunks; c++) {
            receiver_sem.wait_min(c + 1);
            if (c + 1 == num_chunks) {
                push_block();
            }
            if (forward) {
                forward_landed(c + 1);
            }
        }
    }

    if (forward && forwarded < num_chunks) {
        if (!downstream_ready) {
            sender_sem.wait(1);
            sender_sem.set(0);
            downstream_ready = true;
        }
        forward_landed(num_chunks);
    }
}
