// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Standardize - Compute Kernel
// Runs on RISCV_2 (TRISC), performs FPU/SFPU operations
//
// Pipeline for each block (tile-row):
// Phase 1: Tilize: cb_rm_in (32 sticks) -> cb_tilized (Wt tiles)
// Phase 2: Mean reduce: SUM reduce cb_tilized with scaler -> cb_mean (1 tile)
// Phase 3: Subtract: sub<COL> cb_tilized - cb_mean -> cb_xmm (Wt tiles)
// Phase 4: Square: square<> cb_xmm -> cb_xmm_sq (Wt tiles)
// Phase 5: Var reduce: SUM reduce cb_xmm_sq with scaler -> cb_var (1 tile)
// Phase 6: Add+Rsqrt: add_bcast_scalar cb_var + cb_eps -> rsqrt -> cb_invstd (1 tile)
// Phase 7: Normalize: mul<COL> cb_xmm * cb_invstd -> cb_tilized_out (Wt tiles)
// Phase 8: Untilize: cb_tilized_out (Wt tiles) -> cb_rm_out (32 sticks)
//
// Compile-time args:
//   0: Wt - Number of tiles per row (must be constexpr for untilize template)
//   1: nblocks - Total number of tile-rows to process

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t nblocks = get_compile_time_arg_val(1);

    // ========== CB indices ==========
    constexpr uint32_t c_0 = tt::CBIndex::c_0;    // cb_rm_in
    constexpr uint32_t c_1 = tt::CBIndex::c_1;    // cb_scaler
    constexpr uint32_t c_2 = tt::CBIndex::c_2;    // cb_eps
    constexpr uint32_t c_3 = tt::CBIndex::c_3;    // cb_tilized
    constexpr uint32_t c_4 = tt::CBIndex::c_4;    // cb_tilized_out
    constexpr uint32_t c_16 = tt::CBIndex::c_16;  // cb_rm_out
    constexpr uint32_t c_24 = tt::CBIndex::c_24;  // cb_mean
    constexpr uint32_t c_25 = tt::CBIndex::c_25;  // cb_xmm
    constexpr uint32_t c_26 = tt::CBIndex::c_26;  // cb_xmm_sq
    constexpr uint32_t c_27 = tt::CBIndex::c_27;  // cb_var
    constexpr uint32_t c_28 = tt::CBIndex::c_28;  // cb_invstd

    // ========== Hardware initialization ==========
    // Design: compute_kernel_hw_startup(cb_rm_in, cb_scaler, cb_rm_out)
    compute_kernel_hw_startup(c_0, c_1, c_16);

    // ========== Per-block loop ==========
    for (uint32_t block = 0; block < nblocks; ++block) {
        // ============================================================
        // Phase 1: Tilize (RM -> TILE)
        // Design: USE HELPER: compute_kernel_lib::tilize<c_0, c_3>(Wt, 1)
        // Helper handles: wait cb_rm_in(Wt), tilize, push cb_tilized(Wt), pop cb_rm_in(Wt)
        // ============================================================
        compute_kernel_lib::tilize<c_0, c_3>(Wt, 1);

        // ============================================================
        // Phase 2: Mean (SUM reduce * 1/W)
        // Design: USE HELPER: compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>
        // cb_tilized tiles persist (NoPop) for Phase 3. Produces 1 tile in cb_mean.
        // ============================================================
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            c_3, c_1, c_24, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ============================================================
        // Phase 3: Subtract mean (x - mean)
        // Design: USE HELPER: compute_kernel_lib::sub<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile, Bulk>
        // A policy WaitUpfrontPopAtEnd: waits for Wt tiles in cb_tilized, pops all at end -> frees cb_tilized
        // B policy WaitAndPopPerTile: waits/pops cb_mean (1 tile) -> frees cb_mean
        // Output Bulk: reserves Wt upfront, pushes Wt at end into cb_xmm
        // ============================================================
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_3, c_24, c_25, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ============================================================
        // Phase 4: Square (x - mean)^2
        // Design: USE HELPER: compute_kernel_lib::square<WaitUpfrontNoPop, Bulk>
        // Input WaitUpfrontNoPop: cb_xmm tiles persist for Phase 7
        // Output Bulk: pushes Wt tiles to cb_xmm_sq
        // ============================================================
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_25, c_26, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ============================================================
        // Phase 5: Variance (SUM reduce * 1/W)
        // Design: USE HELPER: compute_kernel_lib::reduce<SUM, REDUCE_ROW, BulkWaitBulkPop>
        // Waits for Wt tiles in cb_xmm_sq, processes, pops all. Produces 1 tile in cb_var.
        // ============================================================
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            c_26, c_1, c_27, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ============================================================
        // Phase 6: Add epsilon + rsqrt (NO HELPER - raw tile API)
        // Design: No helper exists for fused add+rsqrt.
        // Manual reconfig, CB management, DST management required.
        // ============================================================

        // Reconfig for add bcast scalar
        reconfig_data_format(c_27, c_2);
        pack_reconfig_data_format(c_28);
        add_bcast_scalar_init_short(c_27, c_2);

        cb_wait_front(c_27, 1);  // variance (1 tile)
        // cb_eps (c_2) is persistent, already available
        cb_reserve_back(c_28, 1);

        tile_regs_acquire();
        add_tiles_bcast_scalar(c_27, c_2, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, c_28);
        tile_regs_release();

        cb_push_back(c_28, 1);
        cb_pop_front(c_27, 1);  // free cb_var
        // NOTE: Do NOT pop cb_eps (c_2) - it is persistent

        // ============================================================
        // Phase 7: Normalize (x - mean) * inv_std
        // Design: USE HELPER: compute_kernel_lib::mul<COL, WaitUpfrontPopAtEnd, WaitAndPopPerTile, Bulk>
        // A policy WaitUpfrontPopAtEnd: waits for cb_xmm (already there from Phase 4 NoPop), pops Wt at end
        // B policy WaitAndPopPerTile: waits/pops cb_invstd (1 tile)
        // Output Bulk: pushes Wt tiles to cb_tilized_out
        // ============================================================
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            c_25, c_28, c_4, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ============================================================
        // Phase 8: Untilize (TILE -> RM)
        // Design: USE HELPER: compute_kernel_lib::untilize<Wt, c_4, c_16>(1)
        // Helper handles: wait cb_tilized_out(Wt), untilize, push cb_rm_out(Wt), pop cb_tilized_out(Wt)
        // ============================================================
        compute_kernel_lib::untilize<Wt, c_4, c_16>(1);
    }
}
