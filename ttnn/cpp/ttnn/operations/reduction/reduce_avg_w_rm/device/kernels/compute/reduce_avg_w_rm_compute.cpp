// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for reduce_avg_w_rm operation
// Performs tilize -> reduce (width) -> untilize on row-major data.
//
// Per Kernel Design Document:
// - Phase 1: Tilize - Convert Wt tiles of row-major data to TILE_LAYOUT
// - Phase 2: Reduce - Sum Wt tiles across width with scaler 1/W to compute average
// - Phase 3: Untilize - Convert 1 reduced tile back to row-major format
//
// IMPORTANT: Uses helper functions that encapsulate all CB operations.
// DO NOT add cb_wait/pop/reserve/push around helper calls.

#include <cstdint>
#include "compute_kernel_api/common.h"

// Define reduce operation type and dimension BEFORE including helpers
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

// Include helper libraries
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace NAMESPACE {

void MAIN {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);  // Width in tiles (input tiles per row)

    // Runtime args
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // CB indices (per design document)
    constexpr uint32_t cb_rm_in = tt::CBIndex::c_0;    // Input row-major sticks
    constexpr uint32_t cb_tilized = tt::CBIndex::c_1;  // Tilized intermediate
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;   // Scaler tile (1/W)
    constexpr uint32_t cb_reduced = tt::CBIndex::c_3;  // Reduced output (1 tile per row)
    constexpr uint32_t cb_rm_out = tt::CBIndex::c_16;  // Output row-major sticks

    // =========================================================================
    // Initialize compute kernel hardware
    // REQUIRED before using any helper functions
    // =========================================================================
    compute_kernel_hw_startup(cb_rm_in, cb_scaler, cb_rm_out);

    // Process each tile row
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        // =====================================================================
        // Phase 1: Tilize
        // USE HELPER: compute_kernel_lib::tilize(cb_rm_in, Wt, cb_tilized, 1)
        // Helper handles: wait(Wt) from cb_rm_in, reserve/push(Wt) to cb_tilized
        // =====================================================================
        compute_kernel_lib::tilize(cb_rm_in, Wt, cb_tilized, 1);

        // =====================================================================
        // Phase 2: Reduce (Width reduction with streaming input)
        // USE HELPER: compute_kernel_lib::reduce<SUM, REDUCE_ROW>(...)
        // Helper handles: streaming wait/pop from cb_tilized, scaler wait,
        //                 DST management, reserve/push to cb_reduced
        // =====================================================================
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_tilized,  // Input CB (tilized data)
            cb_scaler,   // Scaler CB (1/W value)
            cb_reduced,  // Output CB (reduced tile)
            1,           // Ht = 1 (processing one tile row at a time)
            Wt,          // Wt = tiles across width to reduce
            1            // num_batches = 1 (one tile row per iteration)
        );

        // =====================================================================
        // Phase 3: Untilize
        // USE HELPER: compute_kernel_lib::untilize<1, cb_reduced, cb_rm_out>(1)
        // Helper handles: wait(1) from cb_reduced, reserve/push(1) to cb_rm_out,
        //                 pop(1) from cb_reduced
        // =====================================================================
        compute_kernel_lib::untilize<1, cb_reduced, cb_rm_out>(1);
    }
}

}  // namespace NAMESPACE
