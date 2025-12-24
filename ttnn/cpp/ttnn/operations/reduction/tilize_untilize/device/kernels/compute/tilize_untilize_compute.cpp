// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

/**
 * Compute kernel for tilize_untilize operation.
 *
 * Data flow (per block):
 *   CB_in (c_0, row-major) -> tilize -> CB_tiled (c_1, tiled)
 *   CB_tiled (c_1, tiled) -> [identity/placeholder] -> untilize -> CB_out (c_16, row-major)
 *
 * Uses helper libraries to abstract CB operations:
 * - compute_kernel_lib::tilize() handles CB_in -> CB_tiled conversion
 * - compute_kernel_lib::untilize() handles CB_tiled -> CB_out with auto pack_untilize
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
    // CBs can only hold one block worth of data, so we process: tilize -> untilize -> repeat
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Phase 1: Tilize - CB_in (row-major) -> CB_tiled (tiled)
        compute_kernel_lib::tilize(cb_in, num_tiles_per_row, cb_tiled, 1);

        // Phase 2: [PLACEHOLDER FOR MATH OPERATIONS]
        // Future operations would go here, working on tiled data in CB_tiled
        // For template: data passes through unchanged (identity operation)

        // Phase 3: Untilize - CB_tiled (tiled) -> CB_out (row-major)
        // Automatically uses pack_untilize (hardware-accelerated) when tile_width fits in DEST
        compute_kernel_lib::untilize<num_tiles_per_row, cb_tiled, cb_out>(1);
    }
}
}  // namespace NAMESPACE
