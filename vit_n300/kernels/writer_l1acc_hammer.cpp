// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel: reads output tiles from CB and writes to DRAM for host verification.

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t out_cb = tt::CBIndex::c_16;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_size_bytes = get_arg_val<uint32_t>(2);

    // Get DRAM noc coordinates for bank 0
    uint32_t dst_noc_x = get_arg_val<uint32_t>(3);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(4);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(out_cb, 1);

        uint32_t read_addr = get_read_ptr(out_cb);
        uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);

        noc_async_write(read_addr, dst_noc_addr, tile_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(out_cb, 1);
        dst_addr += tile_size_bytes;
    }
}
