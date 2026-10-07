// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer: takes this node's output tiles from DFB out and gets them into the output tensor.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_out(dfb::out);

#if SHARDED
    // The DFB is the output shard: compute already packed the results in place. Wait for all of them so
    // the program does not finish early, then release the slots.
    dfb_out.wait_front(num_tiles);
    dfb_out.pop_front(num_tiles);
#else
    // Interleaved: write each finished tile over the NoC to its place in the output tensor.
    Noc noc;
    const auto out = TensorAccessor(tensor::out);
    const uint32_t out_tile_bytes = dfb_out.get_entry_size();

    for (uint32_t tile = start_tile_id; tile < start_tile_id + num_tiles; ++tile) {
        dfb_out.wait_front(1);
        noc.async_write(dfb_out, out, out_tile_bytes, {}, {.page_id = tile});
        noc.async_write_barrier();
        dfb_out.pop_front(1);
    }
#endif
}
