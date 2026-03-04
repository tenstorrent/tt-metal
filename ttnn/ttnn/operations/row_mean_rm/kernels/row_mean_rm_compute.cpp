// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// row_mean_rm - Compute Kernel
// Tilize input, reduce mean across row, untilize mean tile to output.
//
// Compile-time args:
//   [0] num_rows   - total tile-rows to process (N_outer * Ht)
//   [1] Wt         - tiles per row (W / 32)

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_out_rm = 16;
constexpr uint32_t cb_input_tiled = 24;
constexpr uint32_t cb_mean = 25;

void kernel_main() {
    constexpr uint32_t num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_out_rm);

    // Wait for scaler (loaded once by reader, never popped)
    cb_wait_front(cb_scaler, 1);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Tilize input RM sticks -> tiled format
        compute_kernel_lib::tilize<cb_input_rm, cb_input_tiled>(Wt, 1);

        // Restore HW state after tilize for enforce_fp32_accumulation reduce
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_input_tiled, cb_scaler)));
        MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>(cb_mean)));

        // Reduce sum across row with 1/W scaler = mean
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_input_tiled, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        cb_pop_front(cb_input_tiled, Wt);

        // Untilize the 1-tile mean result to output RM CB
        // untilize with WaitBlock mode handles wait/pop internally
        compute_kernel_lib::untilize<1, cb_mean, cb_out_rm>(1);
    }
}
