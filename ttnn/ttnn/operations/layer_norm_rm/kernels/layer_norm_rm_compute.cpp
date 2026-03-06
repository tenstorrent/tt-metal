// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_gamma_rm = 1;
constexpr uint32_t cb_beta_rm = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_beta = 4;
constexpr uint32_t cb_reduce_scaler = 8;
constexpr uint32_t cb_eps = 9;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_tilized = 24;
constexpr uint32_t cb_mean = 25;
constexpr uint32_t cb_centered = 26;
constexpr uint32_t cb_squared = 27;
constexpr uint32_t cb_var_eps = 28;
constexpr uint32_t cb_inv_std = 29;
constexpr uint32_t cb_normed = 30;
constexpr uint32_t cb_affine_out = 31;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    const uint32_t N = get_arg_val<uint32_t>(0);

    if (N == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_in, cb_reduce_scaler, cb_out);

    // Tilize gamma/beta from RM to tile format (once, before main loop)
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<cb_gamma_rm, cb_gamma>(Wt, 1);
    }
    if constexpr (has_beta) {
        compute_kernel_lib::tilize<cb_beta_rm, cb_beta>(Wt, 1);
    }

    // Wait for constant CBs
    cb_wait_front(cb_reduce_scaler, 1);
    cb_wait_front(cb_eps, 1);
    if constexpr (has_gamma) {
        cb_wait_front(cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        cb_wait_front(cb_beta, Wt);
    }

    for (uint32_t tr = 0; tr < N; tr++) {
        // Phase 1: Tilize input (cb_in -> cb_tilized)
        compute_kernel_lib::tilize<cb_in, cb_tilized>(Wt, 1);

        // Phase 2: Reduce mean (WaitUpfrontNoPop so cb_tilized persists)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: sub<COL> cb_tilized - cb_mean -> cb_centered, pop cb_tilized
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 4: Square centered (WaitUpfrontNoPop so cb_centered persists for Phase 7)
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_squared, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 5: Reduce variance
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_squared, cb_reduce_scaler, cb_var_eps, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Add eps + rsqrt -> cb_inv_std
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_var_eps,
            cb_eps,
            cb_inv_std,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            {},
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: mul<COL> centered * inv_std -> cb_normed, pop cb_centered
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_inv_std, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        if constexpr (has_gamma) {
            // Phase 8a: gamma * normed -> cb_affine_out
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_normed, cb_gamma, cb_affine_out, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

            if constexpr (has_beta) {
                // Phase 8b: (gamma * normed) + beta -> cb_normed (reuse freed CB)
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::ROW,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                    cb_affine_out, cb_beta, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

                // Untilize from cb_normed
                compute_kernel_lib::untilize<Wt, cb_normed, cb_out>(1);
            } else {
                // Untilize from cb_affine_out
                compute_kernel_lib::untilize<Wt, cb_affine_out, cb_out>(1);
            }
        } else if constexpr (has_beta) {
            // Only beta, no gamma: normed + beta -> cb_affine_out
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_normed, cb_beta, cb_affine_out, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

            compute_kernel_lib::untilize<Wt, cb_affine_out, cb_out>(1);
        } else {
            // No affine: untilize normed directly
            compute_kernel_lib::untilize<Wt, cb_normed, cb_out>(1);
        }
    }
}
