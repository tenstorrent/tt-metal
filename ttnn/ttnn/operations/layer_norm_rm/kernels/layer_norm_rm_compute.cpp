// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Runs on math RISC-V core, performs FPU/SFPU operations
//
// This kernel:
// 1. One-time: Tilizes gamma (c_3 -> c_5) and beta (c_4 -> c_6), leaves them persistent
// 2. For each tile-row (10 phases):
//    Phase 1:  Tilize input (c_0 -> c_2)
//    Phase 2:  Mean: reduce<SUM, REDUCE_ROW>(c_2, c_1) -> c_24
//    Phase 3:  Center: sub<COL>(c_2, c_24) -> c_25
//    Phase 4:  Square: square<>(c_25) -> c_26
//    Phase 5:  Variance: reduce<SUM, REDUCE_ROW>(c_26, c_1) -> c_27
//    Phase 6:  Add epsilon + rsqrt: add<SCALAR>(c_27, c_7) + rsqrt -> c_28
//    Phase 7:  Normalize: mul<COL>(c_25, c_28) -> c_29
//    Phase 8:  Apply gamma: mul<NONE>(c_29, c_5) -> c_30
//    Phase 9:  Apply beta: add<NONE>(c_30, c_6) -> c_31
//    Phase 10: Untilize output (c_31 -> c_16)

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

namespace {

// Compile-time arguments
constexpr uint32_t num_tile_rows = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

// CB indices
constexpr uint32_t cb_input_rm = tt::CBIndex::c_0;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_1;
constexpr uint32_t cb_input_tilized = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_3;
constexpr uint32_t cb_beta_rm = tt::CBIndex::c_4;
constexpr uint32_t cb_gamma_tilized = tt::CBIndex::c_5;
constexpr uint32_t cb_beta_tilized = tt::CBIndex::c_6;
constexpr uint32_t cb_eps_scalar = tt::CBIndex::c_7;
constexpr uint32_t cb_output_rm = tt::CBIndex::c_16;
constexpr uint32_t cb_mean = tt::CBIndex::c_24;
constexpr uint32_t cb_centered = tt::CBIndex::c_25;
constexpr uint32_t cb_squared = tt::CBIndex::c_26;
constexpr uint32_t cb_var = tt::CBIndex::c_27;
constexpr uint32_t cb_rstd = tt::CBIndex::c_28;
constexpr uint32_t cb_normalized = tt::CBIndex::c_29;
constexpr uint32_t cb_gamma_applied = tt::CBIndex::c_30;
constexpr uint32_t cb_out_tilized = tt::CBIndex::c_31;

}  // namespace

void kernel_main() {
    // ========== HARDWARE INITIALIZATION ==========
    // Initialize compute hardware with first operation's CBs
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_output_rm);

    // ========== ONE-TIME GAMMA/BETA TILIZE ==========
    // Tilize gamma: c_3 (RM) -> c_5 (tiled, persistent)
    compute_kernel_lib::tilize<cb_gamma_rm, cb_gamma_tilized>(Wt, 1);

    // Tilize beta: c_4 (RM) -> c_6 (tiled, persistent)
    compute_kernel_lib::tilize<cb_beta_rm, cb_beta_tilized>(Wt, 1);

    // c_5 and c_6 persist for the entire program -- never popped

    // ========== PER TILE-ROW LOOP ==========
    for (uint32_t row = 0; row < num_tile_rows; ++row) {
        // ---- Phase 1: Tilize input ----
        // c_0 (RM sticks) -> c_2 (tilized tiles)
        // Helper handles: cb_wait_front(c_0, Wt), tilize_block, cb_pop_front(c_0, Wt), cb_push_back(c_2, Wt)
        compute_kernel_lib::tilize<cb_input_rm, cb_input_tilized>(Wt, 1);

        // ---- Phase 2: Compute mean (reduce SUM row) ----
        // c_2 -> c_24 (1 tile with mean in col0)
        // WaitUpfrontNoPop: waits for all Wt tiles in c_2, does NOT pop (tiles needed for Phase 3)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_input_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ---- Phase 3: Center (x - mean) ----
        // c_2 (input) - c_24 (mean, COL broadcast) -> c_25 (centered)
        // NoWaitPopAtEnd for c_2: tiles already present from Phase 2, pop at end
        // WaitUpfrontNoPop for c_24: already has 1 tile from Phase 2, don't pop yet
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_input_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Manual pop: mean no longer needed
        cb_pop_front(cb_mean, 1);

        // ---- Phase 4: Square centered values ----
        // c_25 -> c_26 ((x - mean)^2)
        // WaitUpfrontNoPop: c_25 tiles needed again in Phase 7
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_centered, cb_squared, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ---- Phase 5: Compute variance (reduce SUM row) ----
        // c_26 -> c_27 (1 tile with variance in col0)
        // WaitAndPopPerTile: stream and pop c_26 tiles
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_squared, cb_reduce_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ---- Phase 6: Add epsilon + rsqrt ----
        // c_27 (var) + c_7 (eps, SCALAR broadcast) -> c_28 (rstd)
        // rsqrt applied as post_op lambda
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
            true>(
            cb_var,
            cb_eps_scalar,
            cb_rstd,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            {},
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // ---- Phase 7: Normalize (centered * rstd) ----
        // c_25 (centered) * c_28 (rstd, COL broadcast) -> c_29 (normalized)
        // NoWaitPopAtEnd for c_25: tiles already present from Phase 3, pop when done
        // WaitUpfrontNoPop for c_28: 1 tile present from Phase 6, don't pop yet
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_centered, cb_rstd, cb_normalized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Manual pop: rstd no longer needed
        cb_pop_front(cb_rstd, 1);

        // ---- Phase 8: Apply gamma (element-wise mul) ----
        // c_29 (normalized) * c_5 (gamma, no broadcast) -> c_30
        // WaitUpfrontPopAtEnd for c_29: all Wt tiles present, pop after processing
        // NoWaitNoPop for c_5: gamma is persistent, already present from setup
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::NONE,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_normalized, cb_gamma_tilized, cb_gamma_applied, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ---- Phase 9: Apply beta (element-wise add) ----
        // c_30 (gamma_applied) + c_6 (beta, no broadcast) -> c_31
        // WaitUpfrontPopAtEnd for c_30: all Wt tiles present, pop after processing
        // NoWaitNoPop for c_6: beta is persistent, already present from setup
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::NONE,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_gamma_applied, cb_beta_tilized, cb_out_tilized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ---- Phase 10: Untilize output ----
        // c_31 (tiled) -> c_16 (RM sticks)
        // Helper handles: cb_wait_front(c_31, Wt), untilize, cb_pop_front(c_31, Wt), cb_push_back(c_16, Wt)
        compute_kernel_lib::untilize<Wt, cb_out_tilized, cb_output_rm>(1);
    }
}
