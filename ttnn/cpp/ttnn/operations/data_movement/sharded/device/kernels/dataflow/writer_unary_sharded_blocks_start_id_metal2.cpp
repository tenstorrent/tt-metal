// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_unary_sharded_blocks_start_id.cpp. Drains a block of tiles out of a DFB and
// scatters it row-by-row into an interleaved output tensor. Only the plumbing changes: the buffer-index
// compile-time arg becomes dfb::out, the accessor-args / base-address pair becomes the tensor::dst
// binding, the positional runtime args become named ones, and the tile size is read off the DFB object
// instead of a free function keyed by buffer id. The transfer loop is untouched.
// Forked rather than converted in place because the legacy file is still bound by factories on the
// legacy positional-arg API.
//
// The binding names below (dfb::out, tensor::dst) and the named argument set are this fork's interface:
// every later consumer inherits them, so they are taken from the kernel's own vocabulary rather than
// any one op's locals, and are not renamed once a consumer exists.

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

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    // dfb::out — the block of output tiles from this core.
    DataflowBuffer dfb_out(dfb::out);

    // single-tile ublocks
    // get_entry_size() (DFB interface entry bytes), not get_tile_size() (descriptor array
    // unpack_tile_size[]): the latter is not arch-portable to Quasar and can be stale on a DM kernel,
    // giving a wrong NOC write size / L1 stride -> stray write. Byte-identical on WH/BH (entry == tile).
    const uint32_t tile_bytes = dfb_out.get_entry_size();

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
