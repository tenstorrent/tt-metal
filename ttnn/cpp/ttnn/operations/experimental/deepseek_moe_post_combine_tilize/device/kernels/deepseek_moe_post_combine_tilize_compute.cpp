// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "tt-metalium/constants.hpp"

void kernel_main() {
    constexpr uint32_t cb_tilize_input_id = get_named_compile_time_arg_val("tilize_input_cb_id");
    constexpr uint32_t cb_tilize_output_id = get_named_compile_time_arg_val("tilize_output_cb_id");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;

    CircularBuffer cb_tilize_input(cb_tilize_input_id);
    CircularBuffer cb_tilize_output(cb_tilize_output_id);

    compute_kernel_hw_startup(cb_tilize_input_id, cb_tilize_output_id);
    fast_tilize_init(cb_tilize_input_id, num_tiles, cb_tilize_output_id);

    cb_tilize_input.wait_front(tile_height);
    cb_tilize_output.reserve_back(num_tiles);

    fast_tilize_block(cb_tilize_input_id, num_tiles, cb_tilize_output_id);

    cb_tilize_output.push_back(num_tiles);
    cb_tilize_input.pop_front(tile_height);

    fast_tilize_uninit(cb_tilize_input_id, cb_tilize_output_id, num_tiles);
}
