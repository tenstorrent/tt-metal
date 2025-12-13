// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes tiles from the output buffer to interleaved dram.
 */

#include "dataflow_api.h"
#include "debug/dprint_pages.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);     // Destination address in dram
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);    // Number of tiles to write
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);  // Tile offset for this core

    constexpr uint32_t blk = get_compile_time_arg_val(0);  // needed for correctness of softmax/LN kernels
    constexpr auto dst_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_out = tt::CBIndex::c_14;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    // DPRINT <<"blk: " <<blk << ENDL();
    DPRINT << "num_tiles: " << num_tiles << ENDL();
    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += 1) {
        for (uint32_t j = 0; j < 1; j++) {
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            // DPRINT <<"Pre wait_Front tile: "<< ENDL();
            //         tt::data_movement::common::print_bf16_pages(l1_read_addr,32,32);
            DPRINT << "tile_id: " << tile_id << ENDL();
            cb_wait_front(cb_out, 1);
            noc_async_write_tile(tile_id, s, l1_read_addr);
            // DPRINT <<"Post wait_Front tile: "<< ENDL();
            //         tt::data_movement::common::print_bf16_pages(l1_read_addr,32,32);
            tile_id++;
            // l1_read_addr += tile_bytes;
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
