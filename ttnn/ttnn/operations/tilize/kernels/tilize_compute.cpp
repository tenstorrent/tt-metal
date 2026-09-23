// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute (TRISC0/1/2) — the `tilize_block` block operation (op_design.md).
//
// ONE compute_kernel_lib::tilize call covers every block of this Tensix core, so
// tilize init / reconfig / uninit happen once per kernel. The helper processes
// core_row_tiles * num_col_blocks helper-blocks of block_width tiles each; every
// quantum is the nominal block_width (the ragged last column block tilizes stale
// tail columns that the writer never writes).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width = get_compile_time_arg_val(2);

    const uint32_t core_row_tiles = get_arg_val<uint32_t>(0);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(1);

    const uint32_t num_col_blocks = (core_col_tiles + block_width - 1) / block_width;
    const uint32_t num_blocks = core_row_tiles * num_col_blocks;

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);
    compute_kernel_lib::tilize<block_width, cb_input_sticks, cb_output_tiles>(num_blocks);
}
