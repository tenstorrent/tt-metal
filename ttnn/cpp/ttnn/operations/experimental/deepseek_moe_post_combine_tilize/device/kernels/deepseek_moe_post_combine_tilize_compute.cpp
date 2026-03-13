// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t tilize_input_cb_id = get_named_compile_time_arg_val("tilize_input_cb_id");
    constexpr uint32_t tilize_output_cb_id = get_named_compile_time_arg_val("tilize_output_cb_id");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

    // Constants
    constexpr uint32_t tile_height = 32;  // TODO: (GR)

    // Setup
    compute_kernel_hw_startup(tilize_input_cb_id, tilize_output_cb_id);
    fast_tilize_init(tilize_input_cb_id, num_tiles, tilize_output_cb_id);

    cb_wait_front(tilize_input_cb_id, tile_height);
    cb_reserve_back(tilize_output_cb_id, num_tiles);

    fast_tilize_block(tilize_input_cb_id, num_tiles, tilize_output_cb_id);

    cb_push_back(tilize_output_cb_id, num_tiles);
    cb_pop_front(tilize_input_cb_id, tile_height);

    fast_tilize_uninit(tilize_input_cb_id, tilize_output_cb_id);
}
