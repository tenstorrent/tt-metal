// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Row-strided variant of writer_unary_interleaved_start_id_blocked.cpp for the
 * distributed layernorm/RMSNorm 2D-core-grid post-all-gather path.
 *
 * Each core owns tiles_per_core_x global rows of row_width tiles. Global rows
 * are Wt_full tiles wide, so after writing one local row the writer must jump
 * row_stride (Wt_full - row_width) to the next owned row. With row_width == 0
 * the writer degrades to flat linear writes (legacy behavior).
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_tiles = get_arg(args::num_tiles);      // Number of tiles to write
    const auto tile_offset = get_arg(args::tile_offset);  // Tile offset for this core
    const auto row_width = get_arg(args::row_width);      // Tiles per owned row (0 = flat)
    const auto row_stride = get_arg(args::row_stride);    // Jump between owned rows

    constexpr auto blk = get_arg(args::blk);  // needed for correctness of softmax/LN kernels

    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    // Destination for the packed output tiles, drained here and written out to the output tensor.
    DataflowBuffer dfb_out_buf(dfb::out);

    const uint32_t tile_bytes = dfb_out_buf.get_tile_size();

    uint32_t tile_id = tile_offset;
    uint32_t in_row = 0;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        dfb_out_buf.wait_front(blk);
        uint32_t write_offset = 0;
        for (uint32_t j = 0; j < blk; j++) {
            noc.async_write(dfb_out_buf, s, tile_bytes, {.offset_bytes = write_offset}, {.page_id = tile_id});
            tile_id++;
            write_offset += tile_bytes;
            if (row_width > 0) {
                in_row++;
                if (in_row == row_width && (i + j + 1) < num_tiles) {
                    tile_id += row_stride;
                    in_row = 0;
                }
            }
        }
        noc.async_write_barrier();
        dfb_out_buf.pop_front(blk);
    }
}
