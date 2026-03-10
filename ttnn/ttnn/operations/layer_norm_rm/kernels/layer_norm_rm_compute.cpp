// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 4 (variance_inv_std): tilize -> reduce_mean -> sub<COL> -> square -> reduce_var -> add_eps+rsqrt ->
// mul_inv_std -> untilize Later stages add: affine (gamma/beta)

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
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
    constexpr uint32_t cb_squared = 5;        // c_5: squared centered tiles
    constexpr uint32_t cb_inv_std = 6;        // c_6: inv_std = rsqrt(var + eps)
    constexpr uint32_t cb_eps = 7;            // c_7: epsilon tile (persistent)
    constexpr uint32_t cb_output_rm = 16;     // c_16: output RM sticks
    constexpr uint32_t cb_normalized = 24;    // c_24: normalized tiles

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

        // Phase 4: Square centered values (c_4 -> c_5)
        // WaitUpfrontNoPop: c_4 persists for Phase 7 (mul_inv_std)
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_centered, cb_squared, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce variance (c_5 -> c_6)
        // SUM with scaler=1/W to compute variance. WaitAndPopPerTile consumes c_5.
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_squared, cb_reduce_scaler, cb_inv_std, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));

        // Phase 6: Add epsilon + rsqrt (c_6 + c_7 -> c_6)
        // c_6 is both input (variance) and output (inv_std).
        // WaitAndPopPerTile pops variance from c_6, then reserves+pushes inv_std back to c_6.
        // c_7 is persistent epsilon tile (WaitUpfrontNoPop).
        // PostOp applies rsqrt after the addition.
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_inv_std, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::of(1, 1), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply by inv_std (c_4, c_6 -> c_24)
        // c_4 already waited from Phase 4 (NoWaitNoPop), c_6 inv_std consumed by WaitAndPopPerTile
        // COL broadcast: c_6 Col0 inv_std is broadcast to all Wt tiles
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_centered, cb_inv_std, cb_normalized, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop c_4 after Phase 7 (was NoWaitNoPop)
        cb_pop_front(cb_centered, Wt);

        // Phase 8 (Stage 4 output): Untilize normalized output (c_24 -> c_16)
        compute_kernel_lib::untilize<Wt, cb_normalized, cb_output_rm>(1);
    }
}
