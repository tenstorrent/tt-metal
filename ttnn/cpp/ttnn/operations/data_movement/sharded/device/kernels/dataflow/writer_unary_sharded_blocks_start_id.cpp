// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t padded_offset = get_arg_val<uint32_t>(3);
    const uint32_t block_width_padded_num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t output_width_tiles = get_arg_val<uint32_t>(5);
    const uint32_t start_id_offset = get_arg_val<uint32_t>(6);
    const uint32_t start_id_base = get_arg_val<uint32_t>(7);

    // compile-time args
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_out(cb_id_out);

    uint32_t row_start_tile_id = start_id_base + start_id_offset;
    dfb_out.wait_front(block_width_padded_num_tiles);
    uint32_t l1_read_offset = 0;
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc.async_write(
                dfb_out, s, tile_bytes, {.offset_bytes = l1_read_offset}, {.page_id = tile_id, .offset_bytes = 0});
            tile_id++;
            l1_read_offset += tile_bytes;
        }
        l1_read_offset += padded_offset;
        row_start_tile_id += output_width_tiles;
    }
    noc.async_write_barrier();
    dfb_out.pop_front(block_width_padded_num_tiles);
}
