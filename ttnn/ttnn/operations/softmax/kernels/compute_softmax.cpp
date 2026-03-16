// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"

// CB indices
constexpr uint32_t c_0 = 0;    // input
constexpr uint32_t c_1 = 1;    // scaler
constexpr uint32_t c_16 = 16;  // output

// Compile-time args
constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t num_rows_or_cols = get_compile_time_arg_val(2);  // per-core work units
constexpr uint32_t dim = get_compile_time_arg_val(3);               // 0 = width, 1 = height
constexpr uint32_t numeric_stable = get_compile_time_arg_val(4);

void kernel_main() {
    // Hardware init
    compute_kernel_hw_startup(c_0, c_1, c_16);

    // For data_pipeline stage: identity copy all tiles from c_0 to c_16
    // dim=-1: each work unit has Wt tiles
    // dim=-2: each work unit has Ht tiles
    constexpr uint32_t tiles_per_work_unit = (dim == 0) ? Wt : Ht;
    const uint32_t total_tiles = num_rows_or_cols * tiles_per_work_unit;

    compute_kernel_lib::copy_tiles(c_0, c_16, total_tiles);
}
