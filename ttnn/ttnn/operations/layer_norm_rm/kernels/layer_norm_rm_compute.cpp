// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 5 (affine): full layer norm with optional gamma/beta affine transform.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

namespace NAMESPACE {

void MAIN {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // ========== Runtime args ==========
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // ========== CB IDs ==========
    constexpr uint32_t cb_input_rm = 0;       // c_0: RM sticks from reader
    constexpr uint32_t cb_tilized = 1;        // c_1: tilized input (2*Wt for reuse)
    constexpr uint32_t cb_gamma_rm = 2;       // c_2: gamma RM sticks from reader
    constexpr uint32_t cb_beta_rm = 3;        // c_3: beta RM sticks from reader
    constexpr uint32_t cb_reduce_scaler = 8;  // c_8: reduce scaler (1/W)
    constexpr uint32_t cb_eps = 9;            // c_9: epsilon constant tile
    constexpr uint32_t cb_out_rm = 16;        // c_16: untilized output for writer
    constexpr uint32_t cb_mean = 24;          // c_24: mean tile (1 tile)
    constexpr uint32_t cb_centered = 25;      // c_25: centered values (2*Wt for reuse)
    constexpr uint32_t cb_sq = 26;            // c_26: squared centered / gamma*norm reuse
    constexpr uint32_t cb_var = 27;           // c_27: variance tile
    constexpr uint32_t cb_rstd = 28;          // c_28: rsqrt(var + eps) tile
    constexpr uint32_t cb_norm = 29;          // c_29: normalized output
    constexpr uint32_t cb_gamma_tilized = 30;  // c_30: tilized gamma (persistent)
    constexpr uint32_t cb_beta_tilized = 31;   // c_31: tilized beta (persistent)

    // Determine which CB to untilize from based on gamma/beta presence
    // Neither: c_29, gamma only: c_26, beta only: c_26, both: c_29
    constexpr uint32_t cb_untilize_src = (has_gamma && has_beta) ? cb_norm
                                         : (has_gamma)           ? cb_sq
                                         : (has_beta)            ? cb_sq
                                                                 : cb_norm;

    // ========== Hardware init ==========
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_out_rm);

    // ========== Phase 0 (once): Tilize gamma/beta ==========
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma_tilized>(1);
    }
    if constexpr (has_beta) {
        compute_kernel_lib::tilize<Wt, cb_beta_rm, cb_beta_tilized>(1);
    }

    // ========== Main loop ==========
    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Phase 1: Tilize c_0 -> c_1 (Wt tiles)
        compute_kernel_lib::tilize<Wt, cb_input_rm, cb_tilized>(1);

        // Phase 2: Reduce row for mean: c_1 -> c_24
        // WaitUpfrontNoPop: c_1 tiles persist for Phase 3
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 3: sub(tilized_input, mean) with COL broadcast -> centered
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop c_1 after sub
        cb_pop_front(cb_tilized, Wt);

        // Phase 4: Square centered values: c_25 -> c_26
        // WaitUpfrontNoPop on c_25: tiles persist for Phase 7
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce row for variance: c_26 -> c_27
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_sq, cb_reduce_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 6: Add epsilon + rsqrt: c_27 + c_9 -> c_28
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_var, cb_eps, cb_rstd, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply centered by rstd: c_25 * c_28 with COL broadcast -> c_29
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_rstd, cb_norm, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop c_25 after mul
        cb_pop_front(cb_centered, Wt);

        // Phase 8 (optional): Multiply by gamma with ROW broadcast
        if constexpr (has_gamma) {
            // A: c_29 (normalized, WaitAndPopPerTile)
            // B: c_30 (tilized gamma, WaitUpfrontNoPop -- persistent)
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_norm, cb_gamma_tilized, cb_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        }

        // Phase 9 (optional): Add beta with ROW broadcast
        if constexpr (has_beta) {
            constexpr uint32_t cb_beta_in = has_gamma ? cb_sq : cb_norm;
            constexpr uint32_t cb_beta_out = (has_gamma) ? cb_norm : cb_sq;
            // A: cb_beta_in (from previous phase, WaitAndPopPerTile)
            // B: c_31 (tilized beta, WaitUpfrontNoPop -- persistent)
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_beta_in, cb_beta_tilized, cb_beta_out, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        }

        // Phase 10: Untilize final result -> c_16
        compute_kernel_lib::untilize<Wt, cb_untilize_src, cb_out_rm>(1);
    }

    // Pop persistent gamma/beta tiles at end of program
    if constexpr (has_gamma) {
        cb_pop_front(cb_gamma_tilized, Wt);
    }
    if constexpr (has_beta) {
        cb_pop_front(cb_beta_tilized, Wt);
    }
}

}  // namespace NAMESPACE
