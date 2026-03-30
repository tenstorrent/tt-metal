// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

/**
 * Optimized reader kernel for post-combine reduce.
 *
 * Reads ROW_MAJOR expert outputs as "fake tiles" (no tilization!) and
 * loads weight scalars from DRAM one expert at a time.
 *
 * For each token:
 *   For each expert (0-7):
 *     1. Load weight scalar for this expert (1 tile)
 *     2. Read expert output (7 "fake tiles" = 7168 elements)
 *     3. Compute kernel processes them (multiply + accumulate)
 *
 * This ordering allows compute to process one expert at a time,
 * minimizing CB memory requirements.
 */

constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;
constexpr uint32_t cb_weights = tt::CBIndex::c_1;

// Compile-time arguments
constexpr bool combine_is_dram = get_compile_time_arg_val(0) == 1;
constexpr bool weight_is_dram = get_compile_time_arg_val(1) == 1;
constexpr uint32_t num_tokens = get_compile_time_arg_val(2);
constexpr uint32_t num_experts = get_compile_time_arg_val(3);
constexpr uint32_t emb_dim = get_compile_time_arg_val(4);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(5);

void kernel_main() {
    // Runtime arguments
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t weight_addr = get_arg_val<uint32_t>(1);
    uint32_t tokens_per_core = get_arg_val<uint32_t>(2);
    uint32_t token_start_idx = get_arg_val<uint32_t>(3);

    // Tile size for "fake tiles" (ROW_MAJOR data treated as tiles)
    constexpr uint32_t tile_size = 2048;  // 32×32 bfloat16 = 2048 bytes

    // Page size for combine output (one row of embedding = emb_dim * 2 bytes)
    constexpr uint32_t combine_page_size = emb_dim * 2;

    // Setup address generator for combine output (ROW_MAJOR)
    const InterleavedAddrGenFast<combine_is_dram> combine_addrg = {
        .bank_base_address = combine_addr, .page_size = combine_page_size, .data_format = DataFormat::Float16_b};

    // Setup address generator for weights (scalar per expert)
    const InterleavedAddrGenFast<weight_is_dram> weight_addrg = {
        .bank_base_address = weight_addr,
        .page_size = tile_size,  // Weight stored as tile (only first element used)
        .data_format = DataFormat::Float16_b};

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Process tokens assigned to this core
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for (uint32_t token_idx = 0; token_idx < tokens_per_core; ++token_idx) {
        uint32_t global_token_idx = token_start_idx + token_idx;

        // DPRINT << "READER: Processing token " << global_token_idx << ENDL();

        // ──────────────────────────────────────────────────────────────────────
        // BULK LOAD: Load ALL expert data for this token at once
        // ──────────────────────────────────────────────────────────────────────
        uint32_t total_expert_tiles = num_experts * emb_dim_tiles;
        cb_reserve_back(cb_combine_input, total_expert_tiles);
        uint32_t cb_write_addr = get_write_ptr(cb_combine_input);

        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            uint32_t expert_page_idx = global_token_idx * num_experts + expert_idx;
            noc_async_read_page(expert_page_idx, combine_addrg, cb_write_addr);
            cb_write_addr += combine_page_size;

            // Debug: Print first expert page index
            // if (expert_idx == 0) {
            //     DPRINT << "  READER: Loading experts starting at page " << expert_page_idx << ENDL();
            // }
        }
        noc_async_read_barrier();
        cb_push_back(cb_combine_input, total_expert_tiles);

        // 1. READER: Print first value of first 2 tiles using DATA macros
        // DPRINT_DATA0(DPRINT << "1.READER[tok=" << global_token_idx << "]: t0="
        //              << TileSlice(cb_combine_input, 0, SliceRange{0,1,1,0,1,1}, true, false)
        //              << " t1=" << TileSlice(cb_combine_input, 1, SliceRange{0,1,1,0,1,1}, true, false) << ENDL());

        // ──────────────────────────────────────────────────────────────────────
        // BULK LOAD: Load ALL weights for this token at once
        // ──────────────────────────────────────────────────────────────────────
        cb_reserve_back(cb_weights, num_experts);
        uint32_t weight_write_addr = get_write_ptr(cb_weights);

        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
            noc_async_read_page(weight_page_idx, weight_addrg, weight_write_addr);
            weight_write_addr += tile_size;
        }
        noc_async_read_barrier();
        cb_push_back(cb_weights, num_experts);

        // DPRINT << "  READER: Loaded " << num_experts << " weights" << ENDL();

        // Compute kernel will now process all experts and accumulate in DST
    }
}
