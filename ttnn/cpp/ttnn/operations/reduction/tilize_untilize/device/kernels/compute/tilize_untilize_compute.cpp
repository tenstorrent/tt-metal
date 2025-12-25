// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

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
    // Future:
    // REDUCE_W_SUM = 1,
    // REDUCE_W_MAX = 2,
    // REDUCE_W_AVG = 3,
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
    constexpr uint32_t cb_in = tt::CBIndex::c_0;     // Row-major input from reader
    constexpr uint32_t cb_tiled = tt::CBIndex::c_1;  // Tiled intermediate
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;  // Scaler for reductions (when needed)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;   // Row-major output to writer

    // Initialize compute kernel hardware for tilize/untilize
    compute_kernel_hw_startup(cb_in, cb_out);

    // Process each block one at a time
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // ========== Phase 1: Tilize ==========
        // CB_in (row-major) -> CB_tiled (tiled)
        compute_kernel_lib::tilize(cb_in, num_tiles_per_row, cb_tiled, 1);

        // ========== Phase 2: Math Operation (selected by OpType) ==========
        if constexpr (op_type == OpType::IDENTITY) {
            // Pass-through: data flows directly from cb_tiled to untilize
            // No operation needed - tilize output goes straight to untilize
        }
        // Future operations would be added here as else if constexpr blocks:
        // else if constexpr (op_type == OpType::REDUCE_W_SUM) { ... }
        // else if constexpr (op_type == OpType::RELU) { ... }

        // ========== Phase 3: Untilize ==========
        // CB_tiled (tiled) -> CB_out (row-major)
        // Automatically uses pack_untilize (hardware-accelerated) when tile_width fits in DEST
        if constexpr (op_type == OpType::IDENTITY) {
            compute_kernel_lib::untilize<num_tiles_per_row, cb_tiled, cb_out>(1);
        }
        // Future: reductions would untilize fewer tiles (e.g., 1 for reduce_w)
        // else if constexpr (op_type == OpType::REDUCE_W_SUM) {
        //     compute_kernel_lib::untilize<1, cb_tiled, cb_out>(1);
        // }
    }
}
}  // namespace NAMESPACE
