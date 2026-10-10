// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tilize with one block (tile_height rows) slice per compute thread ("lane"). Lane t tilizes lane_tiles
// tiles per block from block_rows lane-wide row entries of the strided input DFB. The DFB places lane
// t's entries num_lanes slots apart, so full_ct_dim, which sets the unpacker's row stride, is the
// distance between two of this lane's rows in tiles. Each lane owns one input and one output tile
// counter.

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_lane_blocks = get_arg(args::num_lane_blocks);
    constexpr uint32_t lane_tiles = get_arg(args::lane_tiles);
    constexpr uint32_t full_ct_dim = get_arg(args::full_ct_dim);
    constexpr uint32_t block_rows = get_arg(args::block_rows);
    if (num_lane_blocks == 0) {
        return;
    }

    compute_kernel_hw_startup(dfb::in, dfb::out);
    DataflowBuffer in(dfb::in);
    DataflowBuffer out(dfb::out);
    tilize_init(dfb::in, full_ct_dim, dfb::out);
    for (uint32_t m = 0; m < num_lane_blocks; ++m) {
        in.wait_front(block_rows);
        out.reserve_back(lane_tiles);
        tilize_block(dfb::in, lane_tiles, dfb::out);
        out.push_back(lane_tiles);
        in.pop_front(block_rows);
    }
    tilize_uninit(dfb::in, dfb::out);
}
