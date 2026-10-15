// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize compute, single column block (no ragged tail, no bfp4 round pass, no reorder-free path):
// one compute_kernel_lib::tilize call over this core's tile-rows. Kept separate from
// tilize_compute.cpp so the binary carries only this path: on small (few-tile) cores the kernel
// is fetch/launch-bound and the general kernel's extra code is pure fixed cost (BH, measured).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiles_out = get_compile_time_arg_val(2);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(4);

    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_rm_sticks, cb_tiles_out);
    using namespace compute_kernel_lib::tilize_config;
    for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
        const uint32_t rows =
            (core_rows_tiles - rb_start < block_height_tiles) ? core_rows_tiles - rb_start : block_height_tiles;
        compute_kernel_lib::tilize<
            block_width_tiles,
            cb_rm_sticks,
            cb_tiles_out,
            InitUninitMode::InitAndUninit,
            WaitMode::WaitBlock,
            ReconfigureRegisterDatatypeMode::NoReconfigure,
            Fp32Mode::Lossless>(rows);
    }
}
