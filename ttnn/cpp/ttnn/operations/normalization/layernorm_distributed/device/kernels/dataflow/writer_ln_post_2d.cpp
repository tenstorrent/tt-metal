// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * 2D-core-grid writer for distributed LayerNorm/RMSNorm (bounty #56908).
 * A core at 2D grid position (x,y) owns `rows` (= tiles_per_core_x) row-tiles x `cols_per_row`
 * (= tiles_per_core_y) col-tiles of the global [num_tile_rows, Wt_full] output. Those tiles are
 * strided: consecutive rows are Wt_full tiles apart. The generic linear writer only lands them
 * correctly when cols_per_row == Wt_full (cores_y == 1); when columns are split (cores_y > 1) it
 * spills into neighbouring cores. This writer jumps the full row width between rows.
 */

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_tiles = get_arg(args::num_tiles);        // total tiles this core writes (rows*cols_per_row)
    const auto tile_offset = get_arg(args::tile_offset);    // page id of this core's top-left tile
    const auto Wt_full = get_arg(args::Wt_full);            // global row width in tiles (per-row stride)
    const auto cols_per_row = get_arg(args::cols_per_row);  // tiles per row this core owns (tiles_per_core_y)

    constexpr auto blk = get_arg(args::blk);

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out_buf(dfb::out);
    const uint32_t tile_bytes = dfb_out_buf.get_tile_size();

    const uint32_t rows = num_tiles / cols_per_row;  // tiles_per_core_x
    for (uint32_t r = 0; r < rows; r++) {
        // Global rows are Wt_full tiles apart; jump the full width between rows (#56908).
        uint32_t tile_id = tile_offset + r * Wt_full;
        for (uint32_t c = 0; c < cols_per_row; c += blk) {
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
    }
}
