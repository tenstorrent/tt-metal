// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Compute Kernel
// Stage 4 (scale_shift): Full layer_norm with optional gamma/beta
//   P1: reduce mean, P2: subtract mean, P3: square, P4a: reduce variance,
//   P4b: add eps + rsqrt, P5: multiply by rstd,
//   P5b: multiply by gamma (ROW broadcast), P5c: add beta (ROW broadcast)

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_eps = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_beta = 4;
constexpr uint32_t cb_output = 16;
constexpr uint32_t cb_mean_rstd = 24;
constexpr uint32_t cb_x_minus_mean = 25;
constexpr uint32_t cb_variance = 26;
constexpr uint32_t cb_diff_sq_temp = 27;

void kernel_main() {
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);

    // Wait for persistent CBs
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    if constexpr (has_gamma) {
        cb_wait_front(cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        cb_wait_front(cb_beta, Wt);
    }

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // P1: Mean reduction
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_input, cb_scaler, cb_mean_rstd, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // P2: Subtract mean -> cb_x_minus_mean
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_input, cb_mean_rstd, cb_x_minus_mean, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        cb_pop_front(cb_input, Wt);

        // P3: Square differences
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_x_minus_mean, cb_diff_sq_temp, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // P4a: Variance reduction
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_diff_sq_temp, cb_scaler, cb_variance, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // P4b: Add eps + rsqrt -> cb_mean_rstd (reused as rstd)
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

        // P5: Multiply by rstd
        // Output goes to cb_diff_sq_temp if gamma/beta present, else cb_output
        if constexpr (has_gamma || has_beta) {
            // Output to cb_diff_sq_temp (c_27, free since P4a consumed it)
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                cb_x_minus_mean, cb_mean_rstd, cb_diff_sq_temp, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        } else {
            // Output directly to cb_output
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                cb_x_minus_mean, cb_mean_rstd, cb_output, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // P5b: Scale by gamma (conditional)
        if constexpr (has_gamma) {
            // A: cb_diff_sq_temp (Wt tiles, PopAtEnd frees it)
            // B: cb_gamma (Wt tiles, program-lifetime, NoWaitNoPop)
            // Out: cb_x_minus_mean if beta follows (c_25, free since P5 popped it), else cb_output
            if constexpr (has_beta) {
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::ROW,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                    cb_diff_sq_temp, cb_gamma, cb_x_minus_mean, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
            } else {
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::ROW,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                    cb_diff_sq_temp, cb_gamma, cb_output, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
            }
        }

        // P5c: Add beta (conditional)
        if constexpr (has_beta) {
            // Source depends on whether gamma was applied
            constexpr uint32_t src_cb = has_gamma ? cb_x_minus_mean : cb_diff_sq_temp;
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                src_cb, cb_beta, cb_output, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }
    }
}
