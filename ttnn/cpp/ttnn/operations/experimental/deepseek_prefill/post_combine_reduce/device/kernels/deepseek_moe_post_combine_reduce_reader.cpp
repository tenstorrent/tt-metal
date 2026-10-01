// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
// Number of tile-sized CB pages needed to hold one row of emb_dim elements
// (ceil(emb_dim / 1024) when emb_dim is a multiple of 32 but not 1024).
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
// Raw byte count of one emb_dim row — used for the NoC read so we transfer
// exactly emb_dim bytes and tolerate non-1024-aligned embedding dims.
constexpr uint32_t emb_dim_bytes = get_compile_time_arg_val(2);
// Experts read per NoC barrier: num_experts (all of a token's rows at once) when the factory sized
// c_0 for it, else 1. One barrier per 8 KB row left the reader latency-bound.
constexpr uint32_t experts_per_batch = get_compile_time_arg_val(3);
constexpr auto combine_accessor_args = TensorAccessorArgs<4>();

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_chunks = get_arg_val<uint32_t>(2);

    Noc noc;
    CircularBuffer cb_combine_input(cb_combine_input_id);

    const auto combine_addrg = TensorAccessor(combine_accessor_args, combine_addr);
    const uint32_t input_tile_size = cb_combine_input.get_tile_size();

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            for (uint32_t expert_base = 0; expert_base < num_experts; expert_base += experts_per_batch) {
                cb_combine_input.reserve_back(experts_per_batch * emb_dim_cb_tiles);
                for (uint32_t e = 0; e < experts_per_batch; ++e) {
                    uint32_t expert_page_idx = global_token_idx * num_experts + expert_base + e;
                    noc.async_read(
                        combine_addrg,
                        cb_combine_input,
                        emb_dim_bytes,
                        {.page_id = expert_page_idx},
                        {.offset_bytes = e * emb_dim_cb_tiles * input_tile_size});
                }
                noc.async_read_barrier();
                cb_combine_input.push_back(experts_per_batch * emb_dim_cb_tiles);
            }
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
