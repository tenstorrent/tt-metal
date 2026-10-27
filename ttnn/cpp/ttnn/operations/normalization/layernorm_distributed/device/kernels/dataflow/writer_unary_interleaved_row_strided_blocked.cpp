// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel writes num_rows rows of row_tiles output tiles to interleaved dram, with consecutive rows
 * row_stride tiles apart. row_tiles must be a multiple of blk.
 */

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_rows = get_arg(args::num_rows);
    const auto tile_offset = get_arg(args::tile_offset);

    constexpr auto blk = get_arg(args::blk);
    constexpr auto row_tiles = get_arg(args::row_tiles);
    constexpr auto row_stride = get_arg(args::row_stride);
    static_assert(row_tiles % blk == 0, "row_tiles must be a multiple of blk");

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    // Destination for the packed output tiles, drained here and written out to the output tensor.
    DataflowBuffer dfb_out_buf(dfb::out);

    const uint32_t tile_bytes = dfb_out_buf.get_tile_size();

    uint32_t row_start = tile_offset;
    for (uint32_t row = 0; row < num_rows; ++row) {
        uint32_t tile_id = row_start;
        for (uint32_t i = 0; i < row_tiles; i += blk) {
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
        row_start += row_stride;
    }
}
