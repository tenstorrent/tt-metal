// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_weights = tt::CBIndex::c_1;
constexpr uint32_t cb_output = tt::CBIndex::c_16;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(1);
constexpr auto weight_accessor_args = TensorAccessorArgs<2>();
constexpr auto output_accessor_args = TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();

constexpr uint32_t TOKENS_PER_CORE = 32;

void kernel_main() {
    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t token_start_idx = get_arg_val<uint32_t>(2);

    constexpr uint32_t weight_tile_size = get_tile_size(cb_weights);
    constexpr uint32_t output_tile_size = get_tile_size(cb_output);

    const auto weight_addrg = TensorAccessor(weight_accessor_args, weight_addr);
    const auto output_addrg = TensorAccessor(output_accessor_args, output_addr);

    // Phase 1: Stream one weight per expert per token (matching expert-by-expert compute).
    for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CORE; ++token_idx) {
        uint32_t global_token_idx = token_start_idx + token_idx;

        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            cb_reserve_back(cb_weights, 1);
            uint32_t cb_write_addr = get_write_ptr(cb_weights);

            uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
            noc_async_read_page(weight_page_idx, weight_addrg, cb_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_weights, 1);
        }
    }

    // Phase 2: Write output tiles after compute finishes
    constexpr uint32_t tiles_total = emb_dim_tiles * TOKENS_PER_CORE;
    cb_wait_front(cb_output, tiles_total);

    uint32_t cb_read_addr = get_read_ptr(cb_output);

    uint32_t tile_row = token_start_idx / TOKENS_PER_CORE;
    uint32_t start_tile_idx = tile_row * tiles_total;

    for (uint32_t tile_idx = 0; tile_idx < tiles_total; ++tile_idx) {
        noc_async_write_page(start_tile_idx + tile_idx, output_addrg, cb_read_addr);
        cb_read_addr += output_tile_size;
    }

    noc_async_write_barrier();

    cb_pop_front(cb_output, tiles_total);
}
