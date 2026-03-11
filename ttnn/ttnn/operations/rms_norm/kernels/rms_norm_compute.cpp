// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel
// Stage 2: tilize (RM) + square + reduce_mean

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_x = 24;
constexpr uint32_t cb_xsq = 25;
constexpr uint32_t cb_rms = 26;

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t Wt = get_arg_val<uint32_t>(1);
    const uint32_t origin_w = get_arg_val<uint32_t>(2);

    if (num_rows == 0) {
        return;
    }

    // Determine compute input CB based on layout
#if IS_INPUT_RM
    constexpr uint32_t compute_input_cb = cb_x;
    // For RM: reduce outputs to cb_rms, then untilize to cb_out
    constexpr uint32_t reduce_output_cb = cb_rms;
#else
    constexpr uint32_t compute_input_cb = cb_in;
    // For TILE: reduce outputs directly to cb_out (no untilize needed)
    constexpr uint32_t reduce_output_cb = cb_out;
#endif

    // Hardware startup: configure with first operation's CBs
#if IS_INPUT_RM
    // RM path: first op is tilize(cb_in -> cb_x)
    compute_kernel_hw_startup(cb_in, cb_in, cb_x);
#else
    // TILE path: first op is square(cb_in -> cb_xsq)
    compute_kernel_hw_startup(cb_in, cb_in, cb_xsq);
#endif

    for (uint32_t row = 0; row < num_rows; ++row) {
#if IS_INPUT_RM
        // Phase 1: Tilize RM input -> cb_x
        compute_kernel_lib::tilize<cb_in, cb_x>(Wt, 1);
#endif

        // Phase 2: Square - compute_input_cb -> cb_xsq (streaming, 1 tile at a time)
        compute_kernel_lib::square(compute_input_cb, cb_xsq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 3: Reduce SUM along row (with 1/W scaler = mean)
        // cb_xsq -> reduce_output_cb (1 output tile per row)
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_xsq, cb_scaler, reduce_output_cb, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

#if IS_INPUT_RM
        // Untilize the single reduced tile to cb_out for RM output
        compute_kernel_lib::untilize<1, cb_rms, cb_out>(1);
#endif
    }

    // Cleanup persistent CBs
    cb_pop_front(cb_scaler, 1);
}
