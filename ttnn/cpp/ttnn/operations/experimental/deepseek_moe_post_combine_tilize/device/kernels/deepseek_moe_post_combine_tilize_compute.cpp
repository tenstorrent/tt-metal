// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "tt-metalium/constants.hpp"

void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    DataflowBuffer dfb_tilize_input(dfb::tilize_input);
    DataflowBuffer dfb_tilize_output(dfb::tilize_output);

    compute_kernel_hw_startup(dfb::tilize_input, dfb::tilize_output);
    fast_tilize_init(dfb::tilize_input, num_tiles, dfb::tilize_output);

    dfb_tilize_input.wait_front(tile_height);
    dfb_tilize_output.reserve_back(num_tiles);

    fast_tilize_block(dfb::tilize_input, num_tiles, dfb::tilize_output);

    dfb_tilize_output.push_back(num_tiles);
    dfb_tilize_input.pop_front(tile_height);

    fast_tilize_uninit(dfb::tilize_input, dfb::tilize_output, num_tiles);
}
