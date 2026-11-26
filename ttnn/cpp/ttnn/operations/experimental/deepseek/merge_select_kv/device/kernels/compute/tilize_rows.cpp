// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TILED_OUTPUT: tilizes the 32 kv rows this core gathered (row-major in rm) into one row of tiles, which
// write_tiles stores in the TILE output.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);

    DataflowBuffer rm_dfb(dfb::rm);        // consumer <- merge, 32 rows of Dh, row-major
    DataflowBuffer tiled_dfb(dfb::tiled);  // producer -> write_tiles, Dh / 32 tiles

    compute_kernel_hw_startup(rm_dfb.get_id(), tiled_dfb.get_id());
    tilize_init(rm_dfb.get_id(), tiles_per_row, tiled_dfb.get_id());
    rm_dfb.wait_front(tiles_per_row);
    tiled_dfb.reserve_back(tiles_per_row);
    tilize_block(rm_dfb.get_id(), tiles_per_row, tiled_dfb.get_id());
    tiled_dfb.push_back(tiles_per_row);
    rm_dfb.pop_front(tiles_per_row);
    tilize_uninit(rm_dfb.get_id(), tiled_dfb.get_id());
}
