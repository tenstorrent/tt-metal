// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes tiles from the output buffer to interleaved dram.
 */

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);     // Destination address in dram
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);    // Number of tiles to write
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);  // Tile offset for this core

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(1);  // needed for correctness of softmax/LN kernels

    constexpr uint32_t cb_out = tt::CB::c_out0;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        cb_wait_front(cb_out, blk);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < blk; j++) {
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, blk);
    }
}
