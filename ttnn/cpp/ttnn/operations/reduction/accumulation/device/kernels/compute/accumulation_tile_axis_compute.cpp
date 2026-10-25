// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cumsum.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "../accumulation_common.hpp"

// Cumulative sum along one of the two tile axes, without first permuting the scan axis to dim 0.
//
// The reader delivers num_rows runs of tiles_per_row tiles, each run in scan order. cumsum_tile
// scans every column of a tile top to bottom and carries the column totals into the next tile in
// SFPU registers (first=true starts a new run), so a scan along H needs nothing else. A scan along
// W (SCAN_WIDTH) transposes each tile on unpack, scans it, and transposes it back in Dest: the
// carried totals then belong to the original rows.
//
// Dest is 32-bit, so the running totals are fp32 and only the packed output is rounded.
void kernel_main() {
    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t tiles_per_row = get_arg(args::tiles_per_row);

    DataflowBuffer dfb_in_obj(dfb::in);
    DataflowBuffer dfb_out_obj(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);

    constexpr uint32_t DST_TILE = 0;

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            dfb_in_obj.wait_front(ONE_TILE);
            tile_regs_acquire();
#ifdef SCAN_WIDTH
            transpose_init(dfb::in);
            transpose_tile(dfb::in, FIRST_TILE, DST_TILE);
#else
            copy_init(dfb::in);
            copy_tile(dfb::in, FIRST_TILE, DST_TILE);
#endif
            // The unpack/datacopy init above reprograms the math pipeline, so re-init before every
            // tile. The init only records the SFPU sequence without running it, which leaves the
            // totals carried in SFPU registers from the previous tile intact.
            cumsum_tile_init();
            cumsum_tile(DST_TILE, /*first=*/j == 0);
#ifdef SCAN_WIDTH
            transpose_dest_init<DST_ACCUM_MODE>();
            transpose_dest<DST_ACCUM_MODE>(DST_TILE);
#endif
            tile_regs_commit();
            dfb_in_obj.pop_front(ONE_TILE);

            tile_regs_wait();
            dfb_out_obj.reserve_back(ONE_TILE);
            pack_tile(DST_TILE, dfb::out);
            dfb_out_obj.push_back(ONE_TILE);
            tile_regs_release();
        }
    }
}
