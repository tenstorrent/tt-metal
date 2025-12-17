// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);

    const uint32_t tile_size = get_tile_size(src_cb_id);
    uint64_t local_l1_read_addr = get_noc_addr(input_buffer_address);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(src_cb_id, 1);
        uint32_t src_cb_write_addr = get_write_ptr(src_cb_id);

        noc_async_read(local_l1_read_addr, src_cb_write_addr, tile_size);
        noc_async_read_barrier();

        cb_push_back(src_cb_id, 1);
        local_l1_read_addr += tile_size;
    }
}
