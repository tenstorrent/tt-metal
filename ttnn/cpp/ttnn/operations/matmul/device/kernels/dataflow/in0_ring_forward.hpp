// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One hop of the Metal 2.0 gather_in0 matmul's in0 ring, shared by the ring workers' in0 reader and the
// hop cores that only forward. Every core on the ring binds the same in2 buffer, so in2 sits at one L1
// address on all of them and a shard is written to the next core at the address it was read from.

#pragma once

#include <stdint.h>

#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// Whether the activation's in0 shard for ring position `ring_pos` holds any K tiles. Shards are
// `shard_width_in_tiles` wide and cover K in ring order, so when K does not fill the ring the last
// shards are short or empty.
FORCE_INLINE bool in0_shard_is_empty(uint32_t ring_pos, uint32_t shard_width_in_tiles, uint32_t k_tiles) {
    return ring_pos * shard_width_in_tiles >= k_tiles;
}

// Writes the shard at `addr` to the same address on the next core, unless it is empty, then signals that
// core's ring semaphore. The next core counts shards by the semaphore, so an empty shard is still
// signalled; nothing reads its slot.
FORCE_INLINE void forward_in0_shard(
    const Noc& noc,
    Semaphore<>& signal_sem,
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t shard_size_bytes,
    uint32_t next_core_noc_x,
    uint32_t next_core_noc_y,
    bool shard_is_empty) {
    if (!shard_is_empty) {
        const UnicastEndpoint dst_ep;
        noc.async_write(
            CoreLocalMem<uint32_t>(src_addr),
            dst_ep,
            shard_size_bytes,
            {},
            {.noc_x = next_core_noc_x, .noc_y = next_core_noc_y, .addr = dst_addr});
        // Flush the write before issuing the semaphore increment. The write uses the regular write
        // command buffer while the atomic increment uses the AT command buffer; without this flush the
        // atomic can arrive at the destination before the payload, causing the receiver to read stale
        // data.
        noc.async_writes_flushed();
    }
    signal_sem.up(noc, next_core_noc_x, next_core_noc_y, 1);
}
