// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
//
// Stage 4 (full): tilize, mean, center, square, variance, eps+rsqrt, normalize, gamma, beta, untilize
//
// CB Layout:
//   c_0  (cb_in_rm):    RM input sticks from reader (Wt pages)
//   c_1  (cb_tilized):  Tilized input (Wt tiles)
//   c_2  (cb_scaler):   Reduce scaler 1/W (1 tile, persistent)
//   c_3  (cb_eps):      Epsilon tile (1 tile, persistent)
//   c_4  (cb_gamma):    Tilized gamma (Wt tiles, persistent)
//   c_5  (cb_beta):     Tilized beta (Wt tiles, persistent)
//   c_6  (cb_gamma_rm): RM gamma sticks (Wt pages, consumed once)
//   c_7  (cb_beta_rm):  RM beta sticks (Wt pages, consumed once)
//   c_16 (cb_out):      Untilized output for writer (Wt pages)
//   c_24 (cb_mean):     Mean (1 tile, reused for inv_std)
//   c_25 (cb_centered): Centered = x - mean (Wt tiles)
//   c_26 (cb_inv_std):  Variance after reduce (1 tile scratch)
//   c_27 (cb_normed):   Wt-tile scratch (squared, then normed, then final)
//   c_28 (cb_affine):   Wt-tile scratch (gamma*normed)
//
// Compile-time args:
//   [0] Wt: tiles per row (W / 32)
//   [1] has_gamma
//   [2] has_beta
//
// Runtime args:
//   [0] num_rows_per_core: tile-rows this core processes

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

// CB indices
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_eps = 3;
constexpr uint32_t cb_gamma = 4;
constexpr uint32_t cb_beta = 5;
constexpr uint32_t cb_gamma_rm = 6;
constexpr uint32_t cb_beta_rm = 7;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_mean = 24;
constexpr uint32_t cb_centered = 25;
constexpr uint32_t cb_inv_std = 26;
constexpr uint32_t cb_normed = 27;
constexpr uint32_t cb_affine = 28;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // Runtime args
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(0);

    if (num_rows_per_core == 0) {
        return;
    }

    // Hardware init
    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out);

    // ===== One-time: Tilize gamma and beta from RM CBs =====
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<cb_gamma_rm, cb_gamma>(Wt, 1);
    }
    if constexpr (has_beta) {
        compute_kernel_lib::tilize<cb_beta_rm, cb_beta>(Wt, 1);
    }

    // Per tile-row loop
    for (uint32_t row = 0; row < num_rows_per_core; row++) {
        // ---- Phase 1: Tilize input ----
        // c_0 -> c_1 (Wt tiles)
        compute_kernel_lib::tilize<cb_in_rm, cb_tilized>(Wt, 1);

        // ---- Phase 2: Reduce SUM for mean ----
        // c_1 persists (WaitUpfrontNoPop), scaler 1/W makes SUM into MEAN
        // Output: c_24 (1 tile, mean column vector)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ---- Phase 3: centered = tilized - mean ----
        // c_1: NoWaitPopAtEnd (already waited, free after use)
        // c_24: WaitAndPopPerTile (1 tile, COL broadcast, consumed)
        // Output: c_25 (Wt centered tiles, Bulk)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ---- Phase 4: squared = centered^2 ----
        // c_25: WaitUpfrontNoPop (persist for phase 6)
        // Output: c_27 (Wt squared tiles, Bulk)
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_centered, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ---- Phase 5a: variance = reduce SUM of squared ----
        // c_27: WaitAndPopPerTile (consumed)
        // scaler 1/W applied
        // Output: c_26 (1 tile, variance column vector)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_normed, cb_scaler, cb_inv_std, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ---- Phase 5b: inv_std = rsqrt(variance + epsilon) ----
        // c_26 (variance): WaitAndPopPerTile (1 tile, consumed)
        // c_3 (epsilon): WaitUpfrontNoPop (persistent program-lifetime)
        // Output: c_24 (1 tile, reused - inv_std)
        // Post-op: rsqrt applied in DST before pack
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::NONE,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile>(
            cb_inv_std,
            cb_eps,
            cb_mean,  // reuse c_24 for inv_std output
            compute_kernel_lib::BinaryInputBlockShape::single(),
            {},
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // ---- Phase 6: normed = centered * inv_std ----
        // c_25: NoWaitPopAtEnd (already waited from phase 4, consumed here)
        // c_24: WaitAndPopPerTile (1 tile inv_std, COL broadcast, consumed)
        // Output: c_27 (Wt tiles, reused - normalized)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_centered, cb_mean, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        if constexpr (has_gamma) {
            // ---- Phase 7: affine_out = normed * gamma ----
            // c_27: WaitAndPopPerTile (normed, consumed)
            // c_4: NoWaitNoPop (gamma, persistent)
            // Output: c_28 (Wt tiles)
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryOutputPolicy::PerTile>(
                cb_normed, cb_gamma, cb_affine, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

            if constexpr (has_beta) {
                // ---- Phase 8: result = affine_out + beta ----
                // c_28: WaitAndPopPerTile (gamma*normed, consumed)
                // c_5: NoWaitNoPop (beta, persistent)
                // Output: c_27 (Wt tiles, reused for untilize)
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                    compute_kernel_lib::BinaryOutputPolicy::PerTile>(
                    cb_affine, cb_beta, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
            } else {
                // No beta: output is in cb_affine, need to move to cb_normed for untilize
                // Actually we can untilize from cb_affine directly
                // We'll untilize cb_affine below
            }
        } else if constexpr (has_beta) {
            // No gamma, has beta: result = normed + beta
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryOutputPolicy::PerTile>(
                cb_normed, cb_beta, cb_affine, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // ---- Phase 9: Untilize to output ----
        // Determine which CB holds the final result
        if constexpr (has_gamma && has_beta) {
            // Result is in cb_normed (c_27) after phase 8
            compute_kernel_lib::untilize<Wt, cb_normed, cb_out>(1);
        } else if constexpr (has_gamma && !has_beta) {
            // Result is in cb_affine (c_28) after phase 7
            compute_kernel_lib::untilize<Wt, cb_affine, cb_out>(1);
        } else if constexpr (!has_gamma && has_beta) {
            // Result is in cb_affine (c_28) after add
            compute_kernel_lib::untilize<Wt, cb_affine, cb_out>(1);
        } else {
            // No gamma, no beta: result is in cb_normed (c_27) after phase 6
            compute_kernel_lib::untilize<Wt, cb_normed, cb_out>(1);
        }
    }
}
