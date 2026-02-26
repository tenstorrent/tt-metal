// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // --- Runtime args ---
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t shard_height_tiles = get_arg_val<uint32_t>(2);
    uint32_t shard_width_tiles = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);

    // --- Compile-time args ---
    // TensorAccessorArgs occupies the first N compile-time arg slots.
    constexpr auto src_args = TensorAccessorArgs<0>();

    // --- Setup ---
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    // Create a TensorAccessor for the interleaved input buffer
    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);

    // --- Read all tiles for this core's shard ---
    // Reserve space in the output CB for the entire shard at once.
    // The CB is backed by the sharded output buffer (set_globally_allocated_address),
    // so writing here writes directly into the output tensor's L1 memory.
    uint32_t num_tiles = shard_height_tiles * shard_width_tiles;
    cb_reserve_back(cb_out, num_tiles);
    uint32_t l1_addr = get_write_ptr(cb_out);

    // 2D tile reading: works for HEIGHT, WIDTH, and BLOCK sharding.
    // Each row of tiles in the shard is offset by Nt tiles from the previous.
    for (uint32_t h = 0; h < shard_height_tiles; h++) {
        for (uint32_t w = 0; w < shard_width_tiles; w++) {
            noc_async_read_tile(start_tile_id + h * Nt + w, src, l1_addr);
            l1_addr += tile_bytes;
        }
    }

    // Wait for all NOC reads to complete
    noc_async_read_barrier();

    // Signal that all tiles are ready in the CB
    cb_push_back(cb_out, num_tiles);
}
