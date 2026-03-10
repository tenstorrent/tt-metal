// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 3 (subtract_mean): tilize -> reduce(SUM, REDUCE_ROW, WaitUpfrontNoPop) -> sub<COL> -> untilize
// Later stages add: variance_inv_std, affine

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t nblocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // ========== CB CONSTANTS ==========
    constexpr uint32_t cb_input_rm = 0;       // c_0: input RM sticks
    constexpr uint32_t cb_tilized = 1;        // c_1: tilized input tiles
    constexpr uint32_t cb_reduce_scaler = 2;  // c_2: reduce scaler (1/W)
    constexpr uint32_t cb_mean = 3;           // c_3: mean per row (reduced)
    constexpr uint32_t cb_centered = 4;       // c_4: centered tiles (x - mean)
    constexpr uint32_t cb_output_rm = 16;     // c_16: output RM sticks

    // ========== HW STARTUP ==========
    // 3-arg form: srcA=cb_input_rm, srcB=cb_reduce_scaler, ocb=cb_output_rm
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_output_rm);

    // ========== MAIN LOOP ==========
    for (uint32_t block = 0; block < nblocks_per_core; block++) {
        // Phase 1: Tilize input (c_0 -> c_1)
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Wt, 1);

        // Phase 2: Reduce mean (c_1 -> c_3)
        // SUM with scaler=1/W to compute mean. WaitUpfrontNoPop keeps c_1 for Phase 3.
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));

        // Phase 3: Subtract mean (c_1, c_3 -> c_4)
        // c_1 already waited from Phase 2 (NoWaitNoPop), c_3 mean consumed by WaitAndPopPerTile
        // COL broadcast: c_3 Col0 mean is broadcast to all Wt tiles
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop c_1 after Phase 3 (was NoWaitNoPop)
        cb_pop_front(cb_tilized, Wt);

        // Phase 4 (Stage 3 output): Untilize centered output (c_4 -> c_16)
        compute_kernel_lib::untilize<Wt, cb_centered, cb_output_rm>(1);
    }
}
