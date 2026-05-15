// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_out);

    // bf16 unpack -> bfp8 pack: helper pack-reconfigures via output_cb format.
    compute_kernel_lib::tilize<width_tiles, cb_in, cb_out>(num_blocks);
}
