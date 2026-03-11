// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Tilize, reduce_mean, subtract_mean, square, reduce_var, inv_std, normalize, untilize
// Optional gamma multiply and beta add with ROW broadcast

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);         // tiles per row
    constexpr uint32_t Ht = get_compile_time_arg_val(1);         // total tile-rows
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);  // 1 if gamma provided
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);   // 1 if beta provided

    // ========== CB indices ==========
    constexpr uint32_t cb_in = 0;           // RM sticks from reader
    constexpr uint32_t cb_tilized = 1;      // Tilized input tiles
    constexpr uint32_t cb_mean = 2;         // Row mean
    constexpr uint32_t cb_centered = 3;     // x - mean
    constexpr uint32_t cb_sq = 4;           // (x - mean)^2
    constexpr uint32_t cb_var = 5;          // Row variance
    constexpr uint32_t cb_eps = 6;          // Epsilon constant
    constexpr uint32_t cb_inv_std = 7;      // 1/sqrt(var+eps)
    constexpr uint32_t cb_scaler = 8;       // Reduce scaler (1/W)
    constexpr uint32_t cb_gamma = 9;        // Gamma tiles
    constexpr uint32_t cb_beta = 10;        // Beta tiles
    constexpr uint32_t cb_out = 16;         // Untilized RM output
    constexpr uint32_t cb_normalized = 24;  // Normalized output tiles
    constexpr uint32_t cb_affine_tmp = 25;  // Intermediate for gamma result

    // Determine the final CB to untilize from based on gamma/beta
    // Neither: untilize from cb_normalized
    // Gamma only: untilize from cb_affine_tmp
    // Both: untilize from cb_normalized (beta writes back to cb_normalized)
    constexpr uint32_t cb_untilize_src = (has_gamma && has_beta) ? cb_normalized
                                         : (has_gamma)           ? cb_affine_tmp
                                                                 : cb_normalized;

    // ========== Hardware startup ==========
    // 3-arg form: srcA=cb_in (for tilize), srcB=cb_scaler (for reduce), output=cb_out (for untilize)
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // ========== Wait for program-lifetime CBs ==========
    // Epsilon: wait once, use NoWaitNoPop in phase 6, pop once after all rows
    cb_wait_front(cb_eps, 1);

    // Gamma/beta: wait once if present, use NoWaitNoPop, pop once after all rows
    if constexpr (has_gamma) {
        cb_wait_front(cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        cb_wait_front(cb_beta, Wt);
    }

    // ========== Main loop: process each tile-row ==========
    for (uint32_t ht = 0; ht < Ht; ht++) {
        // Phase 1: Tilize (cb_in -> cb_tilized)
        compute_kernel_lib::tilize<
            cb_in,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 2: Reduce Mean (cb_tilized -> cb_mean via cb_scaler)
        // WaitUpfrontNoPop: tiles persist in cb_tilized for Phase 3
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract Mean (cb_tilized - cb_mean -> cb_centered, COL broadcast)
        // A: cb_tilized already waited from Phase 2, use NoWaitNoPop
        // B: cb_mean freshly pushed, use WaitAndPopPerTile (COL broadcast reuses B tile)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Manual pop cb_tilized after Phase 3 (was NoWaitNoPop)
        cb_pop_front(cb_tilized, Wt);

        // Phase 4: Square (cb_centered^2 -> cb_sq)
        // WaitUpfrontNoPop: cb_centered persists for Phase 7
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_sq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 5: Reduce Variance (cb_sq -> cb_var via cb_scaler)
        // Default policy (WaitAndPopPerTile): cb_sq consumed
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_sq, cb_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Inverse Std = rsqrt(var + eps)
        // A: cb_var (1 tile, WaitAndPopPerTile - consumed)
        // B: cb_eps (1 tile, NoWaitNoPop - persists across all tile-rows)
        // post_op: rsqrt_tile on the sum result
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_var, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Normalize (cb_centered * cb_inv_std -> cb_normalized, COL broadcast)
        // A: cb_centered already waited from Phase 4, use NoWaitNoPop
        // B: cb_inv_std freshly pushed, use WaitAndPopPerTile (COL broadcast)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_inv_std, cb_normalized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Manual pop cb_centered after Phase 7 (was NoWaitNoPop)
        cb_pop_front(cb_centered, Wt);

        // Phase 7.5: Multiply by Gamma (if has_gamma, ROW broadcast)
        if constexpr (has_gamma) {
            // A: cb_normalized (WaitAndPopPerTile - consumed tile by tile)
            // B: cb_gamma (NoWaitNoPop - persists, already waited before loop)
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_normalized, cb_gamma, cb_affine_tmp, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 7.6: Add Beta (if has_beta, ROW broadcast)
        if constexpr (has_beta) {
            // Input is cb_affine_tmp (from gamma), output is cb_normalized
            // A: cb_affine_tmp (WaitAndPopPerTile - consumed)
            // B: cb_beta (NoWaitNoPop - persists, already waited before loop)
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_affine_tmp, cb_beta, cb_normalized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 8: Untilize (cb_untilize_src -> cb_out)
        compute_kernel_lib::untilize<Wt, cb_untilize_src, cb_out>(1);
    }

    // ========== Pop program-lifetime CBs ==========
    cb_pop_front(cb_eps, 1);

    if constexpr (has_gamma) {
        cb_pop_front(cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        cb_pop_front(cb_beta, Wt);
    }
}
