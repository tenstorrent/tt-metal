// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in0_ring_all_gather.cpp, which lives beside it. The Metal
// 2.0 gather_in0 matmul binds this fork; the original serves the legacy MeshWorkload builder and the
// fused reduce-scatter matmul. Until the original is retired, changes to either copy likely belong in
// the other too.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// This fork carries the plain ring only. Every ring core computes, so there are no hop cores; there
// is one weight, so in0 is gathered once; and every in0 shard holds the same K / ring_size unpadded
// tiles, so every shard is forwarded. The legacy kernel keeps the hop-core, multi-weight and
// padded-shard paths for the factories that need them.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto shard_width_in_tiles = get_arg(args::shard_width_in_tiles);
    constexpr auto shard_height_in_tiles = get_arg(args::shard_height_in_tiles);
    constexpr auto ring_size = get_arg(args::ring_size);

    // The core this one forwards to: the previous core in ring order, so that at step s this core
    // holds the shard of ring position (ring_idx + s) % ring_size.
    const uint32_t next_core_noc_x = get_arg(args::next_core_noc_x);
    const uint32_t next_core_noc_y = get_arg(args::next_core_noc_y);

    const Noc noc;
    // Counts the shards the core that forwards to this one has landed in this core's in2.
    Semaphore signal_sem(sem::in0_ring_signal);
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in2(dfb::in2);

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb::in0);
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t shard_size_bytes = shard_size_in_tiles * in0_single_tile_size_bytes;

    // in0 is this core's own shard, already resident: the buffer borrows the activation's L1, so
    // publishing it to compute moves no data.
    dfb_in0.reserve_back(shard_size_in_tiles);
    const uint32_t local_shard_read_addr = dfb_in0.get_write_ptr();
    dfb_in0.push_back(shard_size_in_tiles);

    // in2 holds the other ring_size - 1 shards in arrival order. Every ring core runs the same
    // buffer set, so in2 sits at the same L1 address on each of them, and this core writes the next
    // core's slot for step s at its own slot-s address.
    dfb_in2.reserve_back((ring_size - 1) * shard_size_in_tiles);
    const uint32_t l1_write_addr_in2 = dfb_in2.get_write_ptr();

    for (uint32_t shard_cnt = 0; shard_cnt < ring_size; shard_cnt++) {
        const uint32_t curr_shard_write_addr = l1_write_addr_in2 + (shard_size_bytes * shard_cnt);
        const uint32_t curr_shard_read_addr =
            shard_cnt == 0 ? local_shard_read_addr : l1_write_addr_in2 + (shard_size_bytes * (shard_cnt - 1));

        // Wait for the shard this step forwards to have landed.
        signal_sem.wait_min(shard_cnt);

        // The last shard has gone all the way round: the next core already holds it.
        if (shard_cnt < ring_size - 1) {
            const UnicastEndpoint dst_ep;
            noc.async_write(
                CoreLocalMem<uint32_t>(curr_shard_read_addr),
                dst_ep,
                shard_size_bytes,
                {},
                {.noc_x = next_core_noc_x, .noc_y = next_core_noc_y, .addr = curr_shard_write_addr});
            // Flush the write before issuing the semaphore increment. The write uses the regular
            // write command buffer while the atomic increment uses the AT command buffer; without
            // this flush the atomic can arrive at the destination before the payload, causing the
            // receiver to read stale data.
            noc.async_writes_flushed();

            signal_sem.up(noc, next_core_noc_x, next_core_noc_y, 1);
        }

        if (shard_cnt > 0) {
            dfb_in2.push_back(shard_size_in_tiles);
        }
    }

    noc.async_atomic_barrier();
}
