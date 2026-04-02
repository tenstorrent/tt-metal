// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;
constexpr uint32_t cb_weights = tt::CBIndex::c_1;

constexpr bool combine_is_dram = get_compile_time_arg_val(0) == 1;
constexpr bool weight_is_dram = get_compile_time_arg_val(1) == 1;
constexpr uint32_t num_tokens = get_compile_time_arg_val(2);
constexpr uint32_t num_experts = get_compile_time_arg_val(3);
constexpr uint32_t emb_dim = get_compile_time_arg_val(4);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(5);

constexpr uint32_t TOKENS_PER_CORE = 32;

void kernel_main() {
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t weight_addr = get_arg_val<uint32_t>(1);
    uint32_t token_start_idx = get_arg_val<uint32_t>(2);

    constexpr uint32_t tile_size = 2048;
    constexpr uint32_t combine_page_size = emb_dim * 2;
    constexpr uint32_t weight_page_size = 64;

    const InterleavedAddrGen<combine_is_dram> combine_addrg = {
        .bank_base_address = combine_addr, .page_size = combine_page_size};

    const InterleavedAddrGen<weight_is_dram> weight_addrg = {
        .bank_base_address = weight_addr, .page_size = weight_page_size};

    for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CORE; ++token_idx) {
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
        cb_push_back(cb_combine_input, total_expert_tiles);

        cb_reserve_back(cb_weights, num_experts);
        uint32_t weight_cb_base = get_write_ptr(cb_weights);

        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
            uint32_t tile_addr = weight_cb_base + expert_idx * tile_size;
            noc_async_read_page(weight_page_idx, weight_addrg, tile_addr);
        }
        noc_async_read_barrier();
        cb_push_back(cb_weights, num_experts);
    }
}
