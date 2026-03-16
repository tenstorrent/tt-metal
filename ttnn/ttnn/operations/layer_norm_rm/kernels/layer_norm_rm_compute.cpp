// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 4 (final): tilize + reduce(mean) + sub + square + reduce(var)
//                   + add(eps)+rsqrt + mul(inv_std) + affine(gamma/beta) + untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // ========== CB indices ==========
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_gamma = 1;
    constexpr uint32_t cb_beta = 2;
    constexpr uint32_t cb_gamma_rm = 3;
    constexpr uint32_t cb_beta_rm = 4;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_eps = 9;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_tilized = 24;
    constexpr uint32_t cb_mean = 25;
    constexpr uint32_t cb_centered = 26;
    constexpr uint32_t cb_sq = 27;
    constexpr uint32_t cb_var = 28;
    constexpr uint32_t cb_inv_std = 29;
    constexpr uint32_t cb_norm = 30;

    // Determine the final CB before untilize based on affine configuration.
    // Reuse freed CBs for affine intermediates:
    // - cb_tilized (24) is free after phase 3 (sub pops it)
    // - cb_centered (26) is free after phase 7 (mul inv_std pops it)
    constexpr uint32_t cb_after_gamma = cb_tilized;  // reuse for gamma output
    constexpr uint32_t cb_after_beta = cb_centered;  // reuse for beta output

    // ========== Hardware startup ==========
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // ========== Tilize gamma/beta (once, before main loop) ==========
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma>(1);
    }
    if constexpr (has_beta) {
        compute_kernel_lib::tilize<Wt, cb_beta_rm, cb_beta>(1);
    }

    // ========== Main loop ==========
    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Phase 1: Tilize - cb_in (RM) -> cb_tilized (tiled)
        compute_kernel_lib::tilize<Wt, cb_in, cb_tilized>(1);

        // Phase 2: Reduce row (mean) - cb_tilized -> cb_mean
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 3: Subtract mean (centering) - cb_tilized - cb_mean -> cb_centered
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 4: Square centered values - cb_centered^2 -> cb_sq
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce row (variance) - cb_sq -> cb_var
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_sq, cb_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 6: Add epsilon + rsqrt - cb_var + cb_eps -> cb_inv_std
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_var, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::of(1, 1), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply by inv_std (normalize) - cb_centered * cb_inv_std -> cb_norm
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_inv_std, cb_norm, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 8: Affine transform (conditional)
        if constexpr (has_gamma && has_beta) {
            // Both gamma and beta: norm * gamma -> cb_after_gamma, then + beta -> cb_after_beta
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_norm, cb_gamma, cb_after_gamma, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_after_gamma, cb_beta, cb_after_beta, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            compute_kernel_lib::untilize<Wt, cb_after_beta, cb_out>(1);
        } else if constexpr (has_gamma) {
            // Gamma only: norm * gamma -> cb_after_gamma
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_norm, cb_gamma, cb_after_gamma, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            compute_kernel_lib::untilize<Wt, cb_after_gamma, cb_out>(1);
        } else if constexpr (has_beta) {
            // Beta only: norm + beta -> cb_after_gamma (reuse same temp CB)
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_norm, cb_beta, cb_after_gamma, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            compute_kernel_lib::untilize<Wt, cb_after_gamma, cb_out>(1);
        } else {
            // No affine: untilize cb_norm directly
            compute_kernel_lib::untilize<Wt, cb_norm, cb_out>(1);
        }
    }
}
