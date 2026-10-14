// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute — block operation `tilize_block`.
//
// One compute_kernel_lib::tilize call per block (block_height_tiles tile-rows x width tiles):
// init / uninit / reconfig once per block (Phase 0: once per core). The helper owns all CB
// wait/pop/reserve/push (symmetric tile-sized pages, block width per tile-row). The ragged last
// column block is a second instantiation on its own CB pair.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm_sticks_tail = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tiles_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tiles_out_tail = get_compile_time_arg_val(3);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t last_block_width_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_col_blocks = get_compile_time_arg_val(6);
    constexpr bool has_tail = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t num_full_col_blocks = num_col_blocks - (has_tail ? 1 : 0);

    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_rm_sticks, cb_tiles_out);

    for (uint32_t col_block = 0; col_block < num_full_col_blocks; ++col_block) {
        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rows =
                (core_rows_tiles - rb_start < block_height_tiles) ? core_rows_tiles - rb_start : block_height_tiles;
            compute_kernel_lib::tilize<block_width_tiles, cb_rm_sticks, cb_tiles_out>(rows);
        }
    }

    if constexpr (has_tail) {
        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rows =
                (core_rows_tiles - rb_start < block_height_tiles) ? core_rows_tiles - rb_start : block_height_tiles;
            compute_kernel_lib::tilize<last_block_width_tiles, cb_rm_sticks_tail, cb_tiles_out_tail>(rows);
        }
    }
}
