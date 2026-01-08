// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Writes tiles from output buffer to interleaved DRAM.
 * Kept simple for DIT LayerNorm pre/post all-gather.
 */

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_out = tt::CBIndex::c_14;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        uint32_t tiles_this_block = (i + blk <= num_tiles) ? blk : (num_tiles - i);
        cb_wait_front(cb_out, tiles_this_block);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < tiles_this_block; j++) {
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, tiles_this_block);
    }
}
