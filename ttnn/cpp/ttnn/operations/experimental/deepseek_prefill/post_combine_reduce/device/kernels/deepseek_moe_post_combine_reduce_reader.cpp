// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
// Number of tile-sized CB pages holding this core's column part of one row (ceil(part / 1024)).
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
constexpr auto combine_accessor_args = TensorAccessorArgs<2>();

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_chunks = get_arg_val<uint32_t>(2);
    // This core's column part of each row, [part_offset, part_offset + part_bytes) (raw bytes, so a part that is
    // not 1024-aligned is fine).
    const uint32_t part_offset = get_arg_val<uint32_t>(3);
    const uint32_t part_bytes = get_arg_val<uint32_t>(4);

    Noc noc;
    CircularBuffer cb_combine_input(cb_combine_input_id);

    const auto combine_addrg = TensorAccessor(combine_accessor_args, combine_addr);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            // One token's num_experts slots per batch: all reads in flight, one barrier (the per-slot reads are
            // small, so a barrier each would leave this latency-bound). Compute still pops slot by slot.
            cb_combine_input.reserve_back(num_experts * emb_dim_cb_tiles);
            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                uint32_t expert_page_idx = global_token_idx * num_experts + expert_idx;
                noc.async_read(
                    combine_addrg,
                    cb_combine_input,
                    part_bytes,
                    {.page_id = expert_page_idx, .offset_bytes = part_offset},
                    {.offset_bytes = expert_idx * emb_dim_cb_tiles * cb_combine_input.get_tile_size()});
            }
            noc.async_read_barrier();
            cb_combine_input.push_back(num_experts * emb_dim_cb_tiles);
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
