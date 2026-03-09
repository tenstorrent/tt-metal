// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Compute Kernel
// Stage 3 (normalize): Full normalization without scale/shift
//   P1: reduce mean, P2: subtract mean, P3: square, P4a: reduce variance,
//   P4b: add eps + rsqrt, P5: multiply by rstd

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// Compile-time args
constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

// CB indices
constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_eps = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_beta = 4;
constexpr uint32_t cb_output = 16;
constexpr uint32_t cb_mean_rstd = 24;     // dual-use: mean (P1-P2), rstd (P4b-P5)
constexpr uint32_t cb_x_minus_mean = 25;  // x - mean intermediate
constexpr uint32_t cb_variance = 26;      // intermediate variance
constexpr uint32_t cb_diff_sq_temp = 27;  // dual-use: (x-mean)^2 (P3-P4a), temp_norm (P5)

void kernel_main() {
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);

    // Wait for persistent CBs
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // ============================================================
        // Phase 1: Mean reduction
        //   reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>
        //   cb_input -> cb_mean_rstd (1 tile, mean value)
        //   cb_input tiles persist for P2
        // ============================================================
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_input, cb_scaler, cb_mean_rstd, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ============================================================
        // Phase 2: Subtract mean (x - mean) -> cb_x_minus_mean
        //   sub<COL, NoWaitNoPop, WaitAndPopPerTile>
        //   A: cb_input (already waited, NoPop)
        //   B: cb_mean_rstd (popped by helper)
        // ============================================================
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_input, cb_mean_rstd, cb_x_minus_mean, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Manual pop of cb_input
        cb_pop_front(cb_input, Wt);

        // ============================================================
        // Phase 3: Square differences (x - mean)^2
        //   square<WaitUpfrontNoPop>
        //   cb_x_minus_mean -> cb_diff_sq_temp
        //   cb_x_minus_mean tiles persist for P5
        // ============================================================
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_x_minus_mean, cb_diff_sq_temp, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ============================================================
        // Phase 4a: Variance reduction
        //   reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>
        //   cb_diff_sq_temp -> cb_variance (1 tile)
        //   Scaler = 1/W, so SUM * (1/W) = variance
        //   WaitAndPopPerTile frees cb_diff_sq_temp
        // ============================================================
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_diff_sq_temp, cb_scaler, cb_variance, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ============================================================
        // Phase 4b: Add eps + rsqrt
        //   add<SCALAR>(cb_variance, cb_eps -> cb_mean_rstd)
        //   with rsqrt post-op
        //   cb_variance freed by WaitAndPopPerTile
        //   cb_eps is NoWaitNoPop (program-lifetime, already waited)
        //   Reuses cb_mean_rstd (c_24, free since P2 popped it)
        // ============================================================
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_variance,
            cb_eps,
            cb_mean_rstd,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            {},
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // ============================================================
        // Phase 5: Multiply by rstd
        //   mul<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile>
        //   A: cb_x_minus_mean (persisted from P2, PopAtEnd frees it)
        //   B: cb_mean_rstd (1 tile, rstd, popped per COL row)
        //   Out: cb_output (Wt tiles)
        // ============================================================
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_x_minus_mean, cb_mean_rstd, cb_output, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
    }
}
