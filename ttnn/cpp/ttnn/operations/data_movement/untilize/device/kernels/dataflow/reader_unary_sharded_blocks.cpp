// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Block-by-block reader for sharded inputs.
//
// This unified reader is used for sharded block-reader paths in untilize and supports both:
//   - L1 sharded input
//   - DRAM sharded input
//
// Source addressing is shard-based via TensorAccessor:
//   - Runtime arg `start_shard_id` selects the shard for this core.
//   - Each loop iteration reads one block (`tiles_per_block` pages) from that shard.
//   - TensorAccessor resolves the correct NOC address based on compile-time buffer properties.
//
// The kernel still streams one block at a time into a double-buffered DFB, so the DFB only needs up to
// two blocks rather than an entire shard.
//
// This kernel is used when use_block_reader=true in UntilizeMultiCoreProgramFactory:
//   - Uneven sharding: tensor dims don't evenly divide shard dims
//
// Data flow (block reader):
//   Sharded Source (L1/DRAM) DFB (double-buffered)         Compute
//   +------------------+     +----------+----------+
//   | block 0 (1 row)  | --> | block 0  |          | --> untilize_block()
//   | block 1          | --> |          | block 1  | --> untilize_block()
//   | block 2          | --> | block 2  |          | --> untilize_block()
//   | ...              |     +----------+----------+
//   +------------------+
//
// vs. backed DFB (zero-copy, used for even sharding + pack_untilize):
//   L1 Shard Buffer = DFB (aliased)
//   +------------------+
//   | all blocks       | --> compute reads directly
//   +------------------+
void kernel_main() {
    const auto start_shard_id = get_arg(args::start_shard_id);
    const auto num_blocks = get_arg(args::num_blocks);

    constexpr auto tiles_per_block = get_arg(args::tiles_per_block);
    constexpr uint32_t tile_size_bytes = get_tile_size(dfb::in);
    constexpr uint32_t block_size_bytes = tiles_per_block * tile_size_bytes;
    const auto accessor_src = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in(dfb::in);
    auto shard_pages = accessor_src.shard_pages(start_shard_id);
    auto page_iter = shard_pages.begin();
    for (uint32_t b = 0; b < num_blocks; ++b) {
        dfb_in.reserve_back(tiles_per_block);
        // *page_iter is a tensor_accessor::Page; its noc_traits_t specialization resolves the
        // source NoC address, so a single block (tiles_per_block contiguous pages) is read at once.
        noc.async_read(*page_iter, dfb_in, block_size_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
        page_iter += tiles_per_block;
        noc.async_read_barrier();
        dfb_in.push_back(tiles_per_block);
    }
}
