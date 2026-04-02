// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Minimal repro kernel for the row-major tilize -> reduce handoff.
//
// Output is tiled; only column 0 contains the reduced mean values.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t post_tilize_nops = get_compile_time_arg_val(2);
    constexpr uint32_t insert_tensix_sync = get_compile_time_arg_val(3);

    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_x = 24;

    compute_kernel_hw_startup(cb_rm_in, cb_scaler, cb_x);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        compute_kernel_lib::tilize<width_tiles, cb_rm_in, cb_x>(1);

        if constexpr (post_tilize_nops > 0) {
            for (uint32_t i = 0; i < post_tilize_nops; ++i) {
                TTI_NOP;
            }
        }

        if constexpr (insert_tensix_sync) {
            tensix_sync();
        }

        compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
            cb_x, cb_scaler, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(width_tiles));
    }

    cb_pop_front(cb_scaler, 1);
}
