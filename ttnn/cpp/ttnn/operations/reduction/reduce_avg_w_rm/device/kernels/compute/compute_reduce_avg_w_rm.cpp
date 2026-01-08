// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for reduce_avg_w_rm operation
// Implements three-phase fused pipeline: tilize -> reduce -> untilize
// Using helper library functions that encapsulate CB management and DST register handling

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace NAMESPACE {
void MAIN {
    // Compile-time args
    constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    // Runtime args
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // CB indices (from kernel design document)
    constexpr uint32_t cb_in = tt::CBIndex::c_0;       // Input row-major sticks
    constexpr uint32_t cb_tilized = tt::CBIndex::c_1;  // Intermediate tilized tiles
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;   // Scaler tile (1/W)
    constexpr uint32_t cb_reduced = tt::CBIndex::c_3;  // Reduced tiles (1 per row)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;     // Output row-major sticks

    // REQUIRED: Initialize compute kernel hardware before using any helpers
    // The helpers require this to be called first
    compute_kernel_hw_startup(cb_in, cb_out);

    // Process each block (tile row) through the three-phase pipeline
    // Each block: 32 input sticks -> Wt tilized tiles -> 1 reduced tile -> 32 output sticks

    // Phase 1: Tilize all input blocks
    // Helper encapsulates: tilize_init, cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back, tilize_uninit
    // Input: cb_in gets 1 page (32 sticks) per block from reader
    // Output: cb_tilized gets Wt tiles per block
    compute_kernel_lib::tilize(
        cb_in,       // icb: input CB with row-major sticks
        Wt,          // block_w: tiles per row (output count)
        cb_tilized,  // ocb: output CB for tilized tiles
        num_blocks,  // num_blocks: number of tile rows to process
        1,           // subblock_h: 1 for simple loop pattern
        0,           // old_icb: not used (no DT mode)
        1            // input_count: 1 page (32 sticks) per block from reader
    );

    // Phase 2: Reduce each row of Wt tiles to 1 output tile using SUM with 1/W scaler
    // Helper encapsulates: reduce_init, reconfig_data_format, pack_reconfig_data_format,
    //   cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back,
    //   tile_regs_acquire/commit/wait/release, reduce_tile, pack_tile, reduce_uninit
    // Uses STREAMING mode: tiles arrive one-at-a-time from tilize
    // Input: cb_tilized has Wt tiles per row
    // Input: cb_scaler has 1 tile (1/W value)
    // Output: cb_reduced gets 1 tile per row
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        compute_kernel_lib::ReduceInputMode::STREAMING>(
        cb_tilized,                                             // icb: input CB with tilized tiles
        cb_scaler,                                              // icb_scaler: CB with scaler tile
        cb_reduced,                                             // ocb: output CB for reduced tiles
        compute_kernel_lib::TileShape::grid(num_blocks, Wt, 1)  // shape: num_blocks rows x Wt cols x 1 batch
    );

    // Phase 3: Untilize reduced tiles back to row-major format
    // Helper encapsulates: pack_untilize_init or untilize_init (auto-selected),
    //   cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back,
    //   pack_untilize_block or untilize_block (auto-selected), pack_untilize_uninit or untilize_uninit
    // Output width is 1 tile (32 elements) -> uses fast pack_untilize path
    // Input: cb_reduced has 1 tile per row
    // Output: cb_out gets 1 page (32 sticks of width 32) per row
    compute_kernel_lib::untilize<1, cb_reduced, cb_out>(num_blocks);
}
}  // namespace NAMESPACE
