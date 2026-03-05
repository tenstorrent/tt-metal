// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel (Optimized: asymmetric tilize + CB reuse)
//
// Per tile-row: tilize → mean → sub → square → var → eps+rsqrt → mul_rstd → mul_gamma → add_beta → untilize
// Gamma and beta are tilized once at startup from CB 1/2 into CB 28/29 using asymmetric tilize (1 stick).
// Gamma/beta multiply/add use ROW broadcast to replicate the single populated row across all 32 rows.
//
// CB reuse:
//   CB 24: tilized input (phases 1-3) + normalized output (phases 4,7,8,9,10)
//   CB 25: mean (phases 2-3) + var_sum (phases 5-6)
//
// Compile-time args:
//   0: Ht - total tile rows (N*C*H / 32)
//   1: Wt - tiles per row (W / 32)

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // CB indices
    constexpr uint32_t cb_input_rm = 0;
    constexpr uint32_t cb_gamma_rm = 1;
    constexpr uint32_t cb_beta_rm = 2;
    constexpr uint32_t cb_reduce_scaler = 8;
    constexpr uint32_t cb_eps_scaler = 9;
    constexpr uint32_t cb_output_rm = 16;
    constexpr uint32_t cb_tilized = 24;  // Multi-use: tilized input + normalized
    constexpr uint32_t cb_mean = 25;     // Multi-use: mean + var_sum
    constexpr uint32_t cb_centered = 26;
    constexpr uint32_t cb_gamma_tilized = 28;
    constexpr uint32_t cb_beta_tilized = 29;
    constexpr uint32_t cb_rstd = 30;

    // Hardware startup
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_tilized);

    // Phase 0: Tilize gamma and beta (once at startup, asymmetric: 1 stick)
    // CB 1 (RM gamma, 1 stick) -> CB 28 (tiled gamma, Wt tiles)
    // Only row 0 is populated; ROW broadcast in phases 8/9 replicates it.
    compute_kernel_lib::tilize<cb_gamma_rm, cb_gamma_tilized>(Wt, 1, 1);
    // CB 2 (RM beta, 1 stick) -> CB 29 (tiled beta, Wt tiles)
    compute_kernel_lib::tilize<cb_beta_rm, cb_beta_tilized>(Wt, 1, 1);

    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Phase 1: Tilize input row - CB 0 (RM) -> CB 24 (tiles)
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Wt, 1, 32);

        // Phase 2: Reduce row mean - sum * (1/W)
        // WaitUpfrontNoPop: tiles stay in cb_tilized for subtract phase
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract mean from input - COL broadcast
        // cb_tilized freed here (NoWaitPopAtEnd pops all Wt tiles)
        // cb_mean freed here (WaitAndPopPerTile pops 1 tile)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 4: Square centered values
        // WaitUpfrontNoPop: keep centered for Phase 7
        // Output to cb_tilized (now free after Phase 3 popped it)
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile>(
            cb_centered, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce variance - sum of squares * (1/W)
        // cb_tilized consumed here (WaitAndPopPerTile pops Wt tiles)
        // Output to cb_mean (now free after Phase 3 popped it, reused as var_sum)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Add epsilon + rsqrt
        // cb_mean (as var_sum) consumed here (WaitAndPopPerTile pops 1 tile)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile>(
            cb_mean,
            cb_eps_scaler,
            cb_rstd,
            compute_kernel_lib::BinaryInputBlockShape::of(1, 1),
            {},
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply centered by rstd - COL broadcast
        // cb_centered: NoWaitPopAtEnd (pop after use, frees cb_centered)
        // Output to cb_tilized (free after Phase 5 popped it, reused as normalized)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile>(
            cb_centered, cb_rstd, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 8: Multiply by gamma - ROW broadcast
        // ROW broadcast takes row 0 of gamma tile and replicates across all 32 rows.
        // cb_tilized (as normalized): WaitAndPopPerTile (consume normalized tiles)
        // cb_gamma_tilized: WaitUpfrontNoPop (persistent, no pop)
        // Output to cb_centered (now free after Phase 7 popped it)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile>(
            cb_tilized, cb_gamma_tilized, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 9: Add beta - ROW broadcast
        // ROW broadcast takes row 0 of beta tile and replicates across all 32 rows.
        // cb_centered: WaitAndPopPerTile (consume scaled tiles)
        // cb_beta_tilized: WaitUpfrontNoPop (persistent, no pop)
        // Output to cb_tilized (free after Phase 8 popped it)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile>(
            cb_centered, cb_beta_tilized, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 10: Untilize output - CB 24 -> CB 16
        compute_kernel_lib::untilize<Wt, cb_tilized, cb_output_rm>(1);
    }
}
