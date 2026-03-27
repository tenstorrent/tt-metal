// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

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

        // ──────────────────────────────────────────────────────────────────────
        // For each expert, load weight and expert output
        // ──────────────────────────────────────────────────────────────────────
        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            // ──────────────────────────────────────────────────────────────────
            // 1. Load weight scalar for this (token, expert) pair
            // ──────────────────────────────────────────────────────────────────
            uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;

            cb_reserve_back(cb_weights, 1);
            uint32_t weight_write_addr = get_write_ptr(cb_weights);

            // Read weight tile (only first element will be used for SCALAR broadcast)
            noc_async_read_page(weight_page_idx, weight_addrg, weight_write_addr);
            noc_async_read_barrier();

            cb_push_back(cb_weights, 1);

            // ──────────────────────────────────────────────────────────────────
            // 2. Read expert output as "fake tiles" (ROW_MAJOR → treated as tiles)
            // ──────────────────────────────────────────────────────────────────
            // Expert output for this token starts at:
            // [global_token_idx, expert_idx, 0..emb_dim-1]
            uint32_t expert_output_start_page = global_token_idx * num_experts + expert_idx;

            cb_reserve_back(cb_combine_input, emb_dim_tiles);
            uint32_t cb_write_addr = get_write_ptr(cb_combine_input);

            // Read entire expert output (emb_dim bytes = 7168 bytes)
            // This reads ONE ROW of data (one expert's output for this token)
            noc_async_read_page(expert_output_start_page, combine_addrg, cb_write_addr);
            noc_async_read_barrier();

            cb_push_back(cb_combine_input, emb_dim_tiles);

            // Compute kernel will now:
            // - Multiply these emb_dim_tiles by the weight
            // - Accumulate into output
            // Then pop both CBs, and we load the next expert
        }
    }
}
