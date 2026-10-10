// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilize with one block (tile row) slice per compute thread ("lane"). Lane t untilizes lane_tiles
// tiles per block, one at a time, into lane-wide row entries of the strided output DFB. The DFB places
// lane t's entries num_lanes slots apart, so full_ct_dim, which sets pack_untilize's row stride, is the
// distance between two of this lane's rows in tiles. Each lane owns one output tile counter. Only the
// last block may push fewer than block_rows rows; nothing is packed after it.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_lane_blocks = get_arg(args::num_lane_blocks);
    const uint32_t last_block_rows = get_arg(args::last_block_rows);
    constexpr uint32_t lane_tiles = get_arg(args::lane_tiles);
    constexpr uint32_t full_ct_dim = get_arg(args::full_ct_dim);
    constexpr uint32_t block_rows = get_arg(args::block_rows);
    if (num_lane_blocks == 0) {
        return;
    }

    compute_kernel_hw_startup(dfb::in, dfb::out);
    DataflowBuffer in(dfb::in);
    DataflowBuffer out(dfb::out);
    pack_untilize_init<1, full_ct_dim>(dfb::in, dfb::out);
    for (uint32_t m = 0; m < num_lane_blocks; ++m) {
        out.reserve_back(block_rows);
        for (uint32_t j = 0; j < lane_tiles; ++j) {
            in.wait_front(1);
            pack_untilize_block<1, full_ct_dim>(dfb::in, 1, dfb::out, j);
            in.pop_front(1);
        }
        out.push_back(m == num_lane_blocks - 1 ? last_block_rows : block_rows);
    }
    pack_untilize_uninit(dfb::out);
}
