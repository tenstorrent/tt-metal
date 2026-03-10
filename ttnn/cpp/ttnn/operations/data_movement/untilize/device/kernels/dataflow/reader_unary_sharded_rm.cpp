// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Reader for sharded inputs when CB is NOT backed by buffer
// For sharded buffers, the data is already in local L1 memory
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t src_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    uint32_t l1_read_addr = src_addr;

    // Read all tiles from the local L1 sharded buffer into the CB
    cb_reserve_back(cb_id_in0, num_tiles);
    uint32_t cb_write_addr = get_write_ptr(cb_id_in0);

    // For sharded data, use local L1 to L1 copy
    uint64_t local_l1_read_addr = get_noc_addr(l1_read_addr);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Copy tile from local L1 buffer to CB (both are in L1)
        noc_async_read(local_l1_read_addr, cb_write_addr, tile_size_bytes);
        local_l1_read_addr += tile_size_bytes;
        cb_write_addr += tile_size_bytes;
    }

    noc_async_read_barrier();
    cb_push_back(cb_id_in0, num_tiles);
}
