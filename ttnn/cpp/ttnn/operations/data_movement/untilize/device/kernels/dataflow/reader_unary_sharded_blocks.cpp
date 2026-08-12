// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

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
// The kernel still streams one block at a time into a double-buffered CB, so the CB only needs up to
// two blocks rather than an entire shard.
//
// This kernel is used when use_block_reader=true in UntilizeMultiCoreProgramFactory:
//   - Uneven sharding: tensor dims don't evenly divide shard dims
//
// Data flow (block reader):
//   Sharded Source (L1/DRAM) CB (double-buffered)          Compute
//   +------------------+     +----------+----------+
//   | block 0 (1 row)  | --> | block 0  |          | --> untilize_block()
//   | block 1          | --> |          | block 1  | --> untilize_block()
//   | block 2          | --> | block 2  |          | --> untilize_block()
//   | ...              |     +----------+----------+
//   +------------------+
//
// vs. backed CB (zero-copy, used for even sharding + pack_untilize):
//   L1 Shard Buffer = CB (aliased)
//   +------------------+
//   | all blocks       | --> compute reads directly
//   +------------------+
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_shard_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(1);
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t block_size_bytes = tiles_per_block * tile_size_bytes;
    constexpr auto src_args = TensorAccessorArgs<2>();
    const auto accessor_src = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in(cb_id_in0);
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
