// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes tiles from the output buffer to interleaved dram.
 */

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_tiles = get_arg(args::num_tiles);      // Number of tiles to write
    const auto tile_offset = get_arg(args::tile_offset);  // Tile offset for this core

    constexpr auto blk = get_arg(args::blk);  // needed for correctness of softmax/LN kernels
    constexpr auto Wt = get_arg(args::Wt);  // tiles per local row
    // Full global row width in tiles; equals Wt in the 1D path, larger in the 2D core grid path.
    constexpr auto Wt_full = get_arg(args::Wt_full);

    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    // Destination for the packed output tiles, drained here and written out to the output tensor.
    DataflowBuffer dfb_out_buf(dfb::out);

    const uint32_t tile_bytes = dfb_out_buf.get_tile_size();

    // 2D core grid: write row by row, jumping over the other column-cores' segments at
    // each row end. In the 1D path Wt == Wt_full, so the jump is a no-op and this is
    // bit-identical to the old flat loop.
    uint32_t tile_id = tile_offset;
    for (uint32_t r = 0; r < num_tiles; r += Wt) {
        for (uint32_t i = 0; i < Wt; i += blk) {
            dfb_out_buf.wait_front(blk);
            uint32_t write_offset = 0;
            for (uint32_t j = 0; j < blk; j++) {
                noc.async_write(dfb_out_buf, s, tile_bytes, {.offset_bytes = write_offset}, {.page_id = tile_id});
                tile_id++;
                write_offset += tile_bytes;
            }
            noc.async_write_barrier();
            dfb_out_buf.pop_front(blk);
        }
        tile_id += (Wt_full - Wt);
    }
}
