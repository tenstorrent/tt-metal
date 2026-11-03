// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Interleaved tile reader with one reserve, one read barrier and one push per batch of consecutive tiles,
// instead of per tile. The batch is the consumer's wait unit (a tile-row block), so the DFB is never
// pushed in a shape the compute cannot consume.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tiles_per_batch = get_arg(args::tiles_per_batch);

    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);

    const uint32_t tile_bytes = dfb_in.get_tile_size();

    const auto s = TensorAccessor(tensor::src);

    uint32_t page_id = start_id;
    for (uint32_t done = 0; done < num_tiles;) {
        const uint32_t remaining = num_tiles - done;
        const uint32_t n = remaining < tiles_per_batch ? remaining : tiles_per_batch;
        dfb_in.reserve_back(n);
        // The ring depth is a multiple of the batch, so a full batch never straddles the wrap and the
        // tiles can be laid out by stepping one address; only the final short batch may be smaller.
        uint32_t l1_write_addr = dfb_in.get_write_ptr();
        for (uint32_t t = 0; t < n; ++t) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(s, dst, tile_bytes, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
            l1_write_addr += tile_bytes;
            ++page_id;
        }
        noc.async_read_barrier();
        dfb_in.push_back(n);
        done += n;
    }
}
