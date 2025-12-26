// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Provide default REDUCE_OP/REDUCE_DIM so reduce_helpers.hpp always compiles.
// For IDENTITY operations, these are never used (guarded by if constexpr).
#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#endif
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"

/**
 * Compute kernel for tilize_untilize operation (multi-core version).
 *
 * Data flow (per block):
 *   CB_in (c_0, row-major) -> tilize -> CB_tiled (c_1, tiled)
 *   CB_tiled (c_1, tiled) -> [Phase 2: math operation] -> untilize -> CB_out (c_16, row-major)
 *
 * Operation selection via compile-time OpType argument and if constexpr:
 * - OpType::IDENTITY: Pass-through (tilize -> untilize, no Phase 2 operation)
 * - Future: REDUCE_W_SUM, REDUCE_W_MAX, RELU, etc.
 *
 * Compile-time args:
 *   [0] num_tiles_per_row - Tiles per row (width in tiles)
 *   [1] op_type - Operation type enum value
 *   [2+] Operation-specific args (e.g., packed_scaler for reductions)
 *
 * Runtime args:
 *   [0] num_blocks - Number of tile blocks to process (varies per core for cliff handling)
 */

// Operation type enum - must match host-side definition
enum class OpType : uint32_t {
    IDENTITY = 0,
    REDUCE_W_SUM = 1,
    REDUCE_W_MAX = 2,
    REDUCE_W_AVG = 3,
    // Future:
    // RELU = 4,
};

namespace NAMESPACE {
void MAIN {
    // Compile-time args
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(0);
    constexpr OpType op_type = static_cast<OpType>(get_compile_time_arg_val(1));

    // Runtime args - num_blocks varies per core (full cores vs cliff core)
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // CB indices
    constexpr uint32_t cb_in = tt::CBIndex::c_0;      // Row-major input from reader
    constexpr uint32_t cb_tiled = tt::CBIndex::c_1;   // Tiled intermediate
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;  // Scaler for reductions (when needed)
    constexpr uint32_t cb_reduced = tt::CBIndex::c_3;  // Reduced tile (output of reduce)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;     // Row-major output to writer

    // Initialize compute kernel hardware
    if constexpr (op_type == OpType::IDENTITY) {
        compute_kernel_hw_startup(cb_in, cb_out);
    } else {
        // For reductions: input CB, scaler CB, reduced output CB
        compute_kernel_hw_startup(cb_tiled, cb_scaler, cb_reduced);
    }

    // Process each block one at a time
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // ========== Phase 1: Tilize ==========
        // CB_in (row-major) -> CB_tiled (tiled)
        compute_kernel_lib::tilize(cb_in, num_tiles_per_row, cb_tiled, 1);

        // ========== Phase 2: Math Operation (selected by OpType) ==========
        if constexpr (op_type == OpType::IDENTITY) {
            // Pass-through: data flows directly from cb_tiled to untilize
            // No operation needed - tilize output goes straight to untilize
        } else if constexpr (
            op_type == OpType::REDUCE_W_SUM || op_type == OpType::REDUCE_W_MAX || op_type == OpType::REDUCE_W_AVG) {
            // Reduce all tiles in row to single tile
            // compute_kernel_lib::reduce handles everything:
            // - cb_wait_front for scaler
            // - reduce_init/uninit
            // - tile_regs management
            // - reduce_tile loop
            // - pack_tile to output CB
            compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
                cb_tiled,           // input: num_tiles_per_row tiled tiles
                cb_scaler,          // scaler tile
                cb_reduced,         // output: 1 reduced tile
                1,                  // Ht = 1 (one tile row per block)
                num_tiles_per_row,  // Wt
                1);                 // NC = 1 (one batch per block)
        }

        // ========== Phase 3: Untilize ==========
        // CB_tiled (tiled) -> CB_out (row-major) for IDENTITY
        // CB_reduced (tiled) -> CB_out (row-major) for reductions
        if constexpr (op_type == OpType::IDENTITY) {
            compute_kernel_lib::untilize<num_tiles_per_row, cb_tiled, cb_out>(1);
        } else {
            // Untilize single reduced tile
            compute_kernel_lib::untilize<1, cb_reduced, cb_out>(1);
        }
    }
}
}  // namespace NAMESPACE
