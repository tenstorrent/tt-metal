// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

// Block-by-block reader for sharded inputs. The kernel does a simple local L1 to CB copy (not page-by-page via
// TensorAccessor), so it is not fully using device API 1.1 features. Only cb operations are used.
// Reads tiles from the local L1 shard one block (tile-row) at a time into a double-buffered CB, so the
// CB only needs 2 blocks instead of the entire shard.
//
// This kernel is used when use_block_reader=true in UntilizeMultiCoreProgramFactory:
//   - Uneven sharding: tensor dims don't evenly divide shard dims
//
// Data flow (block reader):
//   L1 Shard Buffer          CB (double-buffered)          Compute
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
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(1);
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t block_size_bytes = tiles_per_block * tile_size_bytes;

    experimental::CircularBuffer cb_in(cb_id_in0);
    uint64_t l1_read_addr = get_noc_addr(src_addr);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_in.reserve_back(tiles_per_block);
        uint32_t cb_write_addr = cb_in.get_write_ptr();
        noc_async_read(l1_read_addr, cb_write_addr, block_size_bytes);
        l1_read_addr += block_size_bytes;
        noc_async_read_barrier();
        cb_in.push_back(tiles_per_block);
    }
}
