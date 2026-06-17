// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Block-by-block reader for sharded inputs (L1 or DRAM). Source addressing is shard-based via
// TensorAccessor: runtime arg `start_shard_id` selects the shard for this core, and each iteration
// reads one block (`tiles_per_block` pages) into the double-buffered DFB.
void kernel_main() {
    const uint32_t start_shard_id = get_arg(args::start_shard_id);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t tiles_per_block = get_arg(args::tiles_per_block);

    const auto accessor_src = TensorAccessor(ta::input);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const uint32_t tile_size_bytes = cb_in.get_tile_size();
    const uint32_t block_size_bytes = tiles_per_block * tile_size_bytes;

    auto shard_pages = accessor_src.shard_pages(start_shard_id);
    auto page_iter = shard_pages.begin();
    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_in.reserve_back(tiles_per_block);
        // *page_iter is a tensor_accessor::Page; its noc_traits_t specialization resolves the
        // source NoC address, so a single block (tiles_per_block contiguous pages) is read at once.
        noc.async_read(*page_iter, cb_in, block_size_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
        page_iter += tiles_per_block;
        noc.async_read_barrier();
        cb_in.push_back(tiles_per_block);
    }
}
