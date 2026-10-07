// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// in0 ring hop for the Metal 2.0 gather_in0 matmul: a core on the ring that computes nothing and only
// passes in0 shards on. Hop cores carry the link from the ring's first core round to its last, as the
// hop-core path of reader_bmm_tile_layout_in0_ring_all_gather.cpp does for the legacy builder.
//
// The binding and argument names below are this kernel's interface: every factory that later binds it
// inherits them and cannot rename them.

#include <stdint.h>

#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/matmul/device/kernels/dataflow/in0_ring_forward.hpp"

void kernel_main() {
    constexpr auto shard_width_in_tiles = get_arg(args::shard_width_in_tiles);
    constexpr auto shard_height_in_tiles = get_arg(args::shard_height_in_tiles);
    constexpr auto ring_size = get_arg(args::ring_size);
    constexpr auto k_tiles = get_arg(args::k_tiles);  // the activation's unpadded K

    // The next hop core, or after the last one, the ring's last core.
    const uint32_t next_core_noc_x = get_arg(args::next_core_noc_x);
    const uint32_t next_core_noc_y = get_arg(args::next_core_noc_y);

    const Noc noc;
    // Counts the shards the core before this one has landed in this core's in2.
    Semaphore signal_sem(sem::in0_ring_signal);
    // in2 is bound here for its address. The core before this one writes slot s of in2, the shard of
    // ring position s, at the address its own slot s has, and this core passes it on from there to the
    // same slot of the next core. Nothing on a hop core reads the shards, so they are never pushed.
    DataflowBuffer dfb_in2(dfb::in2);

    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    const uint32_t shard_size_bytes = shard_size_in_tiles * dfb_in2.get_tile_size();

    dfb_in2.reserve_back((ring_size - 1) * shard_size_in_tiles);
    const uint32_t l1_addr_in2 = dfb_in2.get_write_ptr();

    for (uint32_t slot = 0; slot < ring_size - 1; slot++) {
        signal_sem.wait_min(slot + 1);
        const uint32_t shard_addr = l1_addr_in2 + (shard_size_bytes * slot);
        forward_in0_shard(
            noc,
            signal_sem,
            shard_addr,
            shard_addr,
            shard_size_bytes,
            next_core_noc_x,
            next_core_noc_y,
            in0_shard_is_empty(slot, shard_width_in_tiles, k_tiles));
    }

    noc.async_atomic_barrier();
}
