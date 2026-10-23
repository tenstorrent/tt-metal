// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute side of a Metal 2.0 gather_in0 hop core. A hop core only passes in0 shards on, out of its
// in2 (reader_bmm_tile_layout_in0_ring_hop_metal2.cpp), and in2 has to sit at the ring workers' address
// on it. A local buffer needs a consumer on every core it lives on, and on the workers that consumer is
// compute, so the hop binds this kernel as in2's consumer. It pops each shard the hop reader pushes
// without reading it, so in2 ends the program empty.
//
// The argument names below are this kernel's interface: it takes the hop reader's ring geometry.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto shard_width_in_tiles = get_arg(args::shard_width_in_tiles);
    constexpr auto shard_height_in_tiles = get_arg(args::shard_height_in_tiles);
    constexpr auto ring_size = get_arg(args::ring_size);

    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;

    DataflowBuffer dfb_in2(dfb::in2);
    for (uint32_t slot = 0; slot < ring_size - 1; slot++) {
        dfb_in2.wait_front(shard_size_in_tiles);
        dfb_in2.pop_front(shard_size_in_tiles);
    }
}
