// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Reader kernel for interleaved-to-sharded
// Read tiles from an interleaved DRAM buffer into a sharded L1 output CB.
// Must work for HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TODO: Implement reader kernel
    //
    // Runtime args: src_addr, start_tile_id, shard_height_tiles, shard_width_tiles, Nt
    // Compile-time args: TensorAccessorArgs at index 0
    //
    // 1. Read runtime args and compile-time TensorAccessorArgs
    // 2. Create a TensorAccessor for the interleaved input
    // 3. Reserve CB space for the entire shard (shard_height_tiles * shard_width_tiles tiles)
    // 4. Use a 2D nested loop (h over shard_height_tiles, w over shard_width_tiles)
    //    to read tiles: tile_id = start_tile_id + h * Nt + w
    // 5. noc_async_read_barrier() then cb_push_back()
}
