// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute for the block path of the tiled HC transpose. A block is the 32 input tiles (n, c, ht, wt) for
// c = ct * 32 .. ct * 32 + 31, side by side in dfb::src. Untilizing them as one 32-tile row writes row h of every
// tile into one row-major line ordered [c][w]; that line is output tile (n, h, ct, wt) in row-major form, so
// tilizing each line as a one-tile-wide block yields the 32 output tiles in h order.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t BLOCK_TILES = 32;

    compute_kernel_hw_startup(dfb::src, dfb::mid);
    DataflowBuffer src(dfb::src);
    DataflowBuffer mid(dfb::mid);
    DataflowBuffer out(dfb::out);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        compute_kernel_lib::untilize<
            BLOCK_TILES,
            dfb::src,
            dfb::mid,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(src, mid, 1);

        mid.wait_front(BLOCK_TILES);
        pack_reconfig_data_format(dfb::mid, dfb::out);
        tilize_init(dfb::mid, 1, dfb::out);
        for (uint32_t h = 0; h < BLOCK_TILES; ++h) {
            out.reserve_back(1);
            tilize_block(dfb::mid, 1, dfb::out);
            out.push_back(1);
            mid.pop_front(1);
        }
        tilize_uninit(dfb::mid, dfb::out);
        reconfig_data_format_srca(dfb::mid, dfb::src);
        pack_reconfig_data_format(dfb::out, dfb::mid);
    }
}
