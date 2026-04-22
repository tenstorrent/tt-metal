// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_weights = tt::CBIndex::c_1;
constexpr uint32_t cb_dispatch_table = tt::CBIndex::c_2;
constexpr uint32_t cb_indices = tt::CBIndex::c_3;
constexpr uint32_t cb_output = tt::CBIndex::c_16;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_tiles = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(3);
constexpr uint32_t dispatch_table_aligned_page_size = get_compile_time_arg_val(4);
constexpr uint32_t indices_pages_per_core = get_compile_time_arg_val(5);
constexpr uint32_t indices_page_size = get_compile_time_arg_val(6);
constexpr uint32_t indices_aligned_page_size = get_compile_time_arg_val(7);

constexpr auto weight_accessor_args = TensorAccessorArgs<8>();
constexpr auto output_accessor_args = TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();
constexpr auto dispatch_table_accessor_args =
    TensorAccessorArgs<output_accessor_args.next_compile_time_args_offset()>();
constexpr auto indices_accessor_args =
    TensorAccessorArgs<dispatch_table_accessor_args.next_compile_time_args_offset()>();

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t dispatch_table_addr = get_arg_val<uint32_t>(2);
    uint32_t indices_addr = get_arg_val<uint32_t>(3);
    uint32_t token_start_idx = get_arg_val<uint32_t>(4);
    uint32_t num_chunks = get_arg_val<uint32_t>(5);

    constexpr uint32_t weight_tile_size = get_tile_size(cb_weights);
    constexpr uint32_t output_tile_size = get_tile_size(cb_output);

    const auto weight_addrg = TensorAccessor(weight_accessor_args, weight_addr);
    const auto output_addrg = TensorAccessor(output_accessor_args, output_addr);
    const auto dispatch_table_addrg =
        TensorAccessor(dispatch_table_accessor_args, dispatch_table_addr, dispatch_table_page_size);
    const auto indices_addrg = TensorAccessor(indices_accessor_args, indices_addr, indices_page_size);

    // Pre-load dispatch table into CB (c_2) — read once, used by compute
    cb_reserve_back(cb_dispatch_table, dispatch_table_num_pages);
    uint32_t dispatch_table_write_addr = get_write_ptr(cb_dispatch_table);
    for (uint32_t i = 0; i < dispatch_table_num_pages; i++) {
        noc_async_read_page(i, dispatch_table_addrg, dispatch_table_write_addr + i * dispatch_table_aligned_page_size);
    }
    noc_async_read_barrier();
    cb_push_back(cb_dispatch_table, dispatch_table_num_pages);

    int32_t* dispatch_table = (int32_t*)dispatch_table_write_addr;

    constexpr uint32_t tiles_per_chunk = emb_dim_tiles * TOKENS_PER_CHUNK;
    uint32_t current_token_start = token_start_idx;

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Load this chunk's indices into c_3. Compute pops at end of chunk,
        // freeing the slot for the next iteration's load.
        cb_reserve_back(cb_indices, TOKENS_PER_CHUNK);
        uint32_t indices_write_addr = get_write_ptr(cb_indices);
        for (uint32_t i = 0; i < TOKENS_PER_CHUNK; ++i) {
            noc_async_read_page(
                current_token_start + i, indices_addrg, indices_write_addr + i * indices_aligned_page_size);
        }
        noc_async_read_barrier();
        cb_push_back(cb_indices, TOKENS_PER_CHUNK);

        // Phase 1: stream weights for this chunk's tokens
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = current_token_start + token_idx;

            int32_t* token_indices = (int32_t*)(indices_write_addr + token_idx * indices_aligned_page_size);
            bool has_local = false;
            for (uint32_t k = 0; k < num_experts; ++k) {
                int32_t expert_id = token_indices[k];
                if (dispatch_table[expert_id] != -1) {
                    has_local = true;
                    break;
                }
            }

            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                cb_reserve_back(cb_weights, 1);
                uint32_t cb_write_addr = get_write_ptr(cb_weights);

                bool is_last = (expert_idx == num_experts - 1);
                if (!has_local && is_last) {
                    // No local experts for this token — zero the weight tile so compute's
                    // must_zero_init multiply produces zeros regardless of combine_output.
                    volatile uint32_t* ptr = (volatile uint32_t*)cb_write_addr;
                    for (uint32_t w = 0; w < weight_tile_size / sizeof(uint32_t); w++) {
                        ptr[w] = 0;
                    }
                } else {
                    uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
                    noc_async_read_page(weight_page_idx, weight_addrg, cb_write_addr);
                    noc_async_read_barrier();
                }
                cb_push_back(cb_weights, 1);
            }
        }
        current_token_start += TOKENS_PER_CHUNK;

        // Phase 2: drain the output CB for this chunk and write to DRAM.
        // Must be interleaved per chunk (not batched after all chunks) to avoid
        // output CB filling up and deadlocking compute on the next chunk.
        cb_wait_front(cb_output, tiles_per_chunk);
        uint32_t cb_read_addr = get_read_ptr(cb_output);
        uint32_t chunk_tile_start = (token_start_idx / TOKENS_PER_CHUNK + chunk) * tiles_per_chunk;
        for (uint32_t tile_idx = 0; tile_idx < tiles_per_chunk; ++tile_idx) {
            noc_async_write_page(chunk_tile_start + tile_idx, output_addrg, cb_read_addr);
            cb_read_addr += output_tile_size;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output, tiles_per_chunk);
    }
}
