// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_combine_input = tt::CBIndex::c_0;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
// Number of tile-sized CB pages needed to hold one row of emb_dim elements
// (ceil(emb_dim / 1024) when emb_dim is a multiple of 32 but not 1024).
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
// Raw byte count of one emb_dim row — used for the NoC read so we transfer
// exactly emb_dim bytes and tolerate non-1024-aligned embedding dims.
constexpr uint32_t emb_dim_bytes = get_compile_time_arg_val(2);
constexpr auto combine_accessor_args = TensorAccessorArgs<3>();

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_chunks = get_arg_val<uint32_t>(2);

    const auto combine_addrg = TensorAccessor(combine_accessor_args, combine_addr);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                cb_reserve_back(cb_combine_input, emb_dim_cb_tiles);
                uint32_t cb_write_addr = get_write_ptr(cb_combine_input);

                uint32_t expert_page_idx = global_token_idx * num_experts + expert_idx;
                uint64_t noc_addr = combine_addrg.get_noc_addr(expert_page_idx);
                noc_async_read(noc_addr, cb_write_addr, emb_dim_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_combine_input, emb_dim_cb_tiles);
            }
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
