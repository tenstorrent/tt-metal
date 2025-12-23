// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

/**
 * Compute kernel for tilize_untilize operation.
 *
 * Data flow (per block):
 *   CB_in (c_0, row-major) -> tilize -> CB_tiled (c_1, tiled)
 *   CB_tiled (c_1, tiled) -> [identity/placeholder] -> untilize -> CB_out (c_16, row-major)
 *
 * IMPORTANT: Process one block at a time because CBs are sized for only one block!
 */

namespace NAMESPACE {
void MAIN {
    // Compile-time args
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);         // Number of tile blocks to process
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(1);  // Tiles per row (width in tiles)

    // CB indices
    constexpr uint32_t cb_in = tt::CBIndex::c_0;     // Row-major input from reader
    constexpr uint32_t cb_tiled = tt::CBIndex::c_1;  // Tiled intermediate
    constexpr uint32_t cb_out = tt::CBIndex::c_16;   // Row-major output to writer

    // Initialize compute kernel hardware for tilize/untilize
    compute_kernel_hw_startup(cb_in, cb_out);

    // Process each block one at a time
    // CBs can only hold one block worth of data, so we must do:
    // tilize block -> untilize block -> repeat
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // ========================================
        // Phase 1: Tilize - CB_in (row-major) -> CB_tiled (tiled)
        // ========================================
        tilize_init(cb_in, num_tiles_per_row, cb_tiled);

        cb_wait_front(cb_in, num_tiles_per_row);
        cb_reserve_back(cb_tiled, num_tiles_per_row);

        tilize_block(cb_in, num_tiles_per_row, cb_tiled);

        cb_push_back(cb_tiled, num_tiles_per_row);
        cb_pop_front(cb_in, num_tiles_per_row);

        tilize_uninit(cb_in, cb_tiled);

        // ========================================
        // Phase 2: [PLACEHOLDER FOR MATH OPERATIONS]
        // Future operations would go here, working on tiled data in CB_tiled
        // For template: data passes through unchanged (identity operation)
        // ========================================

        // ========================================
        // Phase 3: Untilize - CB_tiled (tiled) -> CB_out (row-major)
        // ========================================
        untilize_init(cb_tiled);

        cb_wait_front(cb_tiled, num_tiles_per_row);
        cb_reserve_back(cb_out, num_tiles_per_row);

        untilize_block(cb_tiled, num_tiles_per_row, cb_out);

        cb_push_back(cb_out, num_tiles_per_row);
        cb_pop_front(cb_tiled, num_tiles_per_row);

        untilize_uninit(cb_tiled);
    }
}
}  // namespace NAMESPACE
