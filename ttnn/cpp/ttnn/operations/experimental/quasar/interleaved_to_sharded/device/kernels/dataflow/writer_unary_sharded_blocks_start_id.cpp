// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // run-time args
    const uint32_t block_height_tiles = get_arg(args::block_height_tiles);
    const uint32_t block_width_tiles = get_arg(args::block_width_tiles);
    const uint32_t padded_offset = get_arg(args::padded_offset);
    const uint32_t block_width_padded_num_tiles = get_arg(args::block_width_padded_num_tiles);
    const uint32_t output_width_tiles = get_arg(args::output_width_tiles);
    const uint32_t start_id_offset = get_arg(args::start_id_offset);
    const uint32_t start_id_base = get_arg(args::start_id_base);

    // single-tile ublocks
    const uint32_t tile_bytes = DataflowBuffer(dfb::out).get_entry_size();

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    uint32_t row_start_tile_id = start_id_base + start_id_offset;
    cb_out.wait_front(block_width_padded_num_tiles);
    uint32_t l1_read_offset = 0;
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc.async_write(
                cb_out, s, tile_bytes, {.offset_bytes = l1_read_offset}, {.page_id = tile_id, .offset_bytes = 0});
            tile_id++;
            l1_read_offset += tile_bytes;
        }
        l1_read_offset += padded_offset;
        row_start_tile_id += output_width_tiles;
    }
    noc.async_write_barrier();
    cb_out.pop_front(block_width_padded_num_tiles);
}
