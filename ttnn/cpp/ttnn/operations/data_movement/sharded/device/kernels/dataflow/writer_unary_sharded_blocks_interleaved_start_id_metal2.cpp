// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_sharded_blocks_interleaved_start_id.cpp.
//
// Consumes a block of sharded tiles from a DFB and writes them tile-by-tile to
// the interleaved DRAM/L1 output via a TensorAccessor.
//
// Bindings (named, from host KernelSpec):
//   dfb::out                  — DFB endpoint (CONSUMER) — backed by SRC_DFB or OUT_DFB
//   ta::out                   — TensorAccessor (output, interleaved)
//   args::block_height_tiles
//   args::block_width_tiles
//   args::unpadded_block_height_tiles
//   args::unpadded_block_width_tiles
//   args::output_width_tiles
//   args::block_num_tiles
//   args::start_id_offset
//   args::start_id_base

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto block_height_tiles = get_arg(args::block_height_tiles);
    auto block_width_tiles = get_arg(args::block_width_tiles);
    auto unpadded_block_height_tiles = get_arg(args::unpadded_block_height_tiles);
    auto unpadded_block_width_tiles = get_arg(args::unpadded_block_width_tiles);
    auto output_width_tiles = get_arg(args::output_width_tiles);  // input width in tiles - block width in tiles
    auto block_num_tiles = get_arg(args::block_num_tiles);        // block_height_tiles * block_width_tiles
    auto start_id_offset = get_arg(args::start_id_offset);
    auto start_id_base = get_arg(args::start_id_base);
    const uint32_t start_id = start_id_base + start_id_offset;

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(dfb::out);

    const auto s = TensorAccessor(ta::out);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    const uint32_t padded_width_diff = (block_width_tiles - unpadded_block_width_tiles) * tile_bytes;

    uint32_t row_start_tile_id = start_id;
    cb_out.wait_front(block_num_tiles);
    uint32_t l1_read_offset = 0;
    for (uint32_t h = 0; h < unpadded_block_height_tiles; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < unpadded_block_width_tiles; w++) {
            noc.async_write(cb_out, s, tile_bytes, {.offset_bytes = l1_read_offset}, {.page_id = tile_id});
            tile_id++;
            l1_read_offset += tile_bytes;
        }
        l1_read_offset += padded_width_diff;
        row_start_tile_id += output_width_tiles;
    }
    noc.async_write_barrier();
    cb_out.pop_front(block_num_tiles);
}
