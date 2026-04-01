// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Compute Kernel
//
// Per tile-row block (32 rows × Wt tiles):
//   1. Tilize input
//   2. Reduce mean (REDUCE_ROW, SUM with 1/W scaler)
//   3. Sub mean (x - mean, COL broadcast)
//   4. Square (centered^2)
//   5. Reduce var (REDUCE_ROW, SUM with 1/W scaler)
//   6. Add eps + rsqrt → inv_std
//   7. Mul inv_std (centered * inv_std, COL broadcast)
//   8. Mul gamma (ROW broadcast, optional)
//   9. Add beta (ROW broadcast, optional)
//  10. Untilize output

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    // ── Compile-time args ──
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // ── CB indices ──
    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t cb_gamma_rm = 1;
    constexpr uint32_t cb_beta_rm = 2;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_eps = 9;
    constexpr uint32_t cb_rm_out = 16;
    constexpr uint32_t cb_x = 24;
    constexpr uint32_t cb_reduce = 25;
    constexpr uint32_t cb_centered = 26;
    constexpr uint32_t cb_sq = 27;
    constexpr uint32_t cb_inv_std = 28;
    constexpr uint32_t cb_normed = 29;
    constexpr uint32_t cb_gamma_t = 30;
    constexpr uint32_t cb_beta_t = 31;

    // Determine which CB feeds into untilize based on affine config
    constexpr uint32_t num_affine_ops = has_gamma + has_beta;
    constexpr uint32_t cb_final = (num_affine_ops == 1) ? cb_x : cb_normed;

    // ── Hardware initialization ──
    compute_kernel_hw_startup(cb_rm_in, cb_scaler, cb_x);

    // ── One-time setup: tilize gamma (asymmetric, 1 row) ──
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma_t>(1, 1);
    }

    // ── One-time setup: tilize beta (asymmetric, 1 row) ──
    if constexpr (has_beta) {
        compute_kernel_lib::tilize<Wt, cb_beta_rm, cb_beta_t>(1, 1);
    }

    // ── Main loop: process each tile-row block ──
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Phase 1: Tilize input (symmetric, tile-sized pages)
        compute_kernel_lib::tilize<Wt, cb_rm_in, cb_x>(1);

        // Phase 2: Reduce mean — sum(x) * (1/W)
        // WaitUpfrontNoPop: tiles persist in cb_x for sub phase
        compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            cb_x, cb_scaler, cb_reduce, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Sub mean — x - mean (COL broadcast)
        // A(cb_x): NoWaitPopAtEnd — already waited by reduce, pop at end
        // B(cb_reduce): WaitAndPopPerTile — wait for 1 mean tile, pop after use
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_x, cb_reduce, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 4: Square — centered^2
        // WaitUpfrontNoPop: centered tiles persist for mul inv_std
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_sq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 5: Reduce variance — sum(centered^2) * (1/W)
        // Default WaitAndPopPerTile — consumes cb_sq tiles
        compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
            cb_sq, cb_scaler, cb_reduce, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Add eps + rsqrt — inv_std = rsqrt(var + eps)
        // A(cb_reduce): WaitAndPopPerTile — consumed
        // B(cb_eps): WaitUpfrontNoPop — persists across all blocks
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_reduce, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Mul inv_std — centered * inv_std (COL broadcast)
        // A(cb_centered): NoWaitPopAtEnd — already waited by square, pop at end
        // B(cb_inv_std): WaitAndPopPerTile — consumed
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_inv_std, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 8: Mul gamma (optional, ROW broadcast)
        // A(cb_normed): WaitAndPopPerTile — consumed
        // B(cb_gamma_t): WaitUpfrontNoPop — persists across all blocks
        if constexpr (has_gamma) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_normed, cb_gamma_t, cb_x, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 9: Add beta (optional, ROW broadcast)
        // Input/output CBs depend on whether gamma was applied
        if constexpr (has_beta) {
            constexpr uint32_t cb_in9 = has_gamma ? cb_x : cb_normed;
            constexpr uint32_t cb_out9 = has_gamma ? cb_normed : cb_x;
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_in9, cb_beta_t, cb_out9, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 10: Untilize to row-major output
        compute_kernel_lib::untilize<Wt, cb_final, cb_rm_out>(1);
    }

    // ── End-of-kernel cleanup: pop persistent CBs ──
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
    if constexpr (has_gamma) {
        cb_pop_front(cb_gamma_t, Wt);
    }
    if constexpr (has_beta) {
        cb_pop_front(cb_beta_t, Wt);
    }
}
