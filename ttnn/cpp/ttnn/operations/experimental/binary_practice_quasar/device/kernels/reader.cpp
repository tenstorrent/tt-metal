// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader: makes this node's tiles of one input available to the compute kernel in its DFB. The factory runs it
// twice, each on its own DM core: dfb::a / tensor::a are bound to input a for one instance and to b for the other.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_a(dfb::a);

#if SHARDED
    // The DFB is the shard itself, already in this node's L1: nothing to read, only announce it.
    dfb_a.reserve_back(num_tiles);
    dfb_a.push_back(num_tiles);
#else
    // Interleaved: read each tile over the NoC into a free DFB slot, then announce it.
    Noc noc;
    const auto a = TensorAccessor(tensor::a);

#ifdef ARCH_QUASAR
    // Implicit sync: each read claims a free slot itself and publishes it to compute when it lands.
    for (uint32_t tile = start_tile_id; tile < start_tile_id + num_tiles; ++tile) {
        noc.async_read<NocOptions::TXN_ID>(a, dfb_a, {.page_id = tile}, {});
    }
    // Wait until every read has landed and been published.
    dfb_a.finish();
#else
    const uint32_t a_tile_bytes = dfb_a.get_entry_size();

    for (uint32_t tile = start_tile_id; tile < start_tile_id + num_tiles; ++tile) {
        dfb_a.reserve_back(1);
        noc.async_read(a, dfb_a, a_tile_bytes, {.page_id = tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_a.push_back(1);
    }
#endif
#endif
}
