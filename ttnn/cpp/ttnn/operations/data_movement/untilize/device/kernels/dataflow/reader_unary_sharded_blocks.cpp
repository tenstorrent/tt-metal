// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Block-by-block reader for sharded inputs (tensor accessor mode).
// Reads tiles from the local L1 shard one block (tile-row) at a time into
// a double-buffered CB, so the CB only needs 2 blocks instead of the entire shard.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(1);
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t block_size_bytes = tiles_per_block * tile_size_bytes;

    uint64_t l1_read_addr = get_noc_addr(src_addr);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_reserve_back(cb_id_in0, tiles_per_block);
        uint32_t cb_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read(l1_read_addr, cb_write_addr, block_size_bytes);
        l1_read_addr += block_size_bytes;
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, tiles_per_block);
    }
}
