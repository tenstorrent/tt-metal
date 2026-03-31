// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

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
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t weight_addr = get_arg_val<uint32_t>(1);
    uint32_t tokens_per_core = get_arg_val<uint32_t>(2);
    uint32_t token_start_idx = get_arg_val<uint32_t>(3);

    constexpr uint32_t tile_size = 2048;

    constexpr uint32_t combine_page_size = emb_dim * 2;
    constexpr uint32_t weight_page_size = 64;  // DRAM alignment forces 64-byte pages (weight is in first 2 bytes)

    const InterleavedAddrGen<combine_is_dram> combine_addrg = {
        .bank_base_address = combine_addr,
        .page_size = combine_page_size};  // InterleavedAddrGen uses page_size for addressing

    const InterleavedAddrGen<weight_is_dram> weight_addrg = {
        .bank_base_address = weight_addr,
        .page_size = weight_page_size};  // InterleavedAddrGen uses page_size for addressing

    // DPRINT_DATA0(DPRINT << "combine_is_dram = " << combine_is_dram << ENDL());
    // DPRINT_DATA0(DPRINT << "weight_is_dram = " << weight_is_dram << ENDL());
    // DPRINT_DATA0(DPRINT << "num_tokens = " << num_tokens << ENDL());
    // DPRINT_DATA0(DPRINT << "num_experts = " << num_experts << ENDL());
    // DPRINT_DATA0(DPRINT << "emb_dim = " << emb_dim << ENDL());
    // DPRINT_DATA0(DPRINT << "emb_dim_tiles = " << emb_dim_tiles << ENDL());
    // DPRINT_DATA0(DPRINT << "combine_addr = " << combine_addr << ENDL());
    // DPRINT_DATA0(DPRINT << "weight_addr = " << weight_addr << ENDL());
    // DPRINT_DATA0(DPRINT << "tokens_per_core = " << tokens_per_core << ENDL());
    // DPRINT_DATA0(DPRINT << "token_start_idx = " << token_start_idx << ENDL());

    SliceRange sr = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1};

    for (uint32_t token_idx = 0; token_idx < tokens_per_core; ++token_idx) {
        uint32_t global_token_idx = token_start_idx + token_idx;

        uint32_t total_expert_tiles = num_experts * emb_dim_tiles;
        cb_reserve_back(cb_combine_input, total_expert_tiles);
        uint32_t cb_write_addr = get_write_ptr(cb_combine_input);

        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            uint32_t expert_page_idx = global_token_idx * num_experts + expert_idx;
            noc_async_read_page(expert_page_idx, combine_addrg, cb_write_addr);
            cb_write_addr += combine_page_size;
        }
        noc_async_read_barrier();
        DPRINT_DATA0(DPRINT << "reader input data -- " << "\n" << ENDL());
        for (uint32_t j = 0; j < total_expert_tiles; j++) {
            DPRINT_DATA0(
                DPRINT << "tile " << j << " values = " << TileSlice(cb_combine_input, j, sr, true, false) << ENDL());
        }
        DPRINT_DATA0(DPRINT << ENDL());
        cb_push_back(cb_combine_input, total_expert_tiles);

        cb_reserve_back(cb_weights, num_experts);
        uint32_t weight_cb_base = get_write_ptr(cb_weights);

        // Read each expert weight (one page = one bf16 weight)
        // Write directly to tile positions (first element of each tile)
        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
            uint32_t tile_addr = weight_cb_base + expert_idx * tile_size;
            noc_async_read_page(weight_page_idx, weight_addrg, tile_addr);
        }
        noc_async_read_barrier();

        cb_push_back(cb_weights, num_experts);

        DPRINT_DATA0(DPRINT << "reader weight data (token " << global_token_idx << "):" << ENDL());
        for (uint32_t j = 0; j < num_experts; j++) {
            DPRINT_DATA0(DPRINT << "  expert " << j << " = " << TileSlice(cb_weights, j, sr, true, false) << ENDL());
        }
        DPRINT_DATA0(DPRINT << ENDL());
    }
}
