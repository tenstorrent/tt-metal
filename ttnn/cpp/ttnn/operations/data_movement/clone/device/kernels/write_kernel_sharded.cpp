// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(0);

    const uint32_t tile_size = get_tile_size(dst_cb_id);
    uint64_t local_l1_write_addr = get_noc_addr(output_buffer_address);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(dst_cb_id, 1);
        uint32_t dst_cb_read_addr = get_read_ptr(dst_cb_id);

        noc_async_write(dst_cb_read_addr, local_l1_write_addr, tile_size);
        noc_async_write_barrier();

        cb_pop_front(dst_cb_id, 1);
        local_l1_write_addr += tile_size;
    }
}
