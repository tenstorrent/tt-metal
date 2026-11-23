// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Interleaved tile reader with one reserve, one read barrier and one push per batch of consecutive tiles,
// instead of per tile. A batch never crosses the end of a tile-row block, the consumer's wait unit: the last
// batch of a block is short when the batch does not divide it.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tiles_per_batch = get_arg(args::tiles_per_batch);
    constexpr uint32_t tiles_per_block = get_arg(args::tiles_per_block);

    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);

    const uint32_t tile_bytes = dfb_in.get_tile_size();

    const auto s = TensorAccessor(tensor::src);

    uint32_t page_id = start_id;
    for (uint32_t done = 0; done < num_tiles;) {
        uint32_t left_in_block = num_tiles - done < tiles_per_block ? num_tiles - done : tiles_per_block;
        done += left_in_block;
        while (left_in_block != 0) {
            const uint32_t n = left_in_block < tiles_per_batch ? left_in_block : tiles_per_batch;
            dfb_in.reserve_back(n);
            // The ring holds whole blocks and a batch ends at a block boundary at the latest, so a batch never
            // straddles the ring wrap and its tiles can be laid out by stepping one address.
            uint32_t l1_write_addr = dfb_in.get_write_ptr();
            for (uint32_t t = 0; t < n; ++t) {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(s, dst, tile_bytes, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
                l1_write_addr += tile_bytes;
                ++page_id;
            }
            noc.async_read_barrier();
            dfb_in.push_back(n);
            left_in_block -= n;
        }
    }
}
