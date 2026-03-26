// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Sum - Compute Kernel
// Performs REDUCE_ROW (SUM) with optional tilize (RM input) and untilize (RM output).
// All 4 layout combinations are handled via compile-time branching.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr bool input_is_rm = get_compile_time_arg_val(2);
    constexpr bool output_is_rm = get_compile_time_arg_val(3);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_tilized = tt::CBIndex::c_24;
    constexpr auto cb_reduced = tt::CBIndex::c_25;

    // new llk

    // Initialize compute hardware
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    for (uint32_t ht = 0; ht < Ht; ht++) {
        if constexpr (input_is_rm && output_is_rm) {
            // RM -> RM: tilize -> reduce -> untilize
            compute_kernel_lib::tilize<Wt, cb_in, cb_tilized>(1);
            compute_kernel_lib::tilize<Wt, cb_in, cb_tilized>(1);
            // for(int i = 0; i < 10; i++){
            //     TTI_NOP;
            // }
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_tilized, cb_scaler, cb_reduced, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));
            compute_kernel_lib::untilize<1, cb_reduced, cb_out>(1);

            // he he ha

        } else if constexpr (input_is_rm && !output_is_rm) {
            // RM -> Tile: tilize -> reduce
            compute_kernel_lib::tilize<Wt, cb_in, cb_tilized>(1);
            // for(int i = 0; i < 10; i++){
            // TTI_NOP;
            // }
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_tilized, cb_scaler, cb_out, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));

        } else if constexpr (!input_is_rm && output_is_rm) {
            // Tile -> RM: reduce -> untilize
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_in, cb_scaler, cb_reduced, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));
            compute_kernel_lib::untilize<1, cb_reduced, cb_out>(1);

        } else {
            // Tile -> Tile: reduce only
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_in, cb_scaler, cb_out, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));
        }
    }
}
