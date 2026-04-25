// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_weights = tt::CBIndex::c_1;
constexpr uint32_t cb_dispatch_table = tt::CBIndex::c_2;
constexpr uint32_t cb_indices = tt::CBIndex::c_3;
constexpr uint32_t cb_output = tt::CBIndex::c_16;

// Fixed CT arg layout for both paths. The dispatch_table / indices metadata
// and accessor args are always emitted by the program factory; when the
// GPT-OSS (weight==0) path is selected, the metadata values are zero and
// the corresponding TensorAccessorArgs reuse the weight tensor's layout so
// the CT arg vector has a deterministic size. use_dispatch_table_skip is
// appended last to guard the runtime code paths that would consult the
// unused slots.
constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_out_tiles = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(3);
constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(4);
constexpr uint32_t dispatch_table_aligned_page_size = get_compile_time_arg_val(5);
constexpr uint32_t indices_pages_per_core = get_compile_time_arg_val(6);
constexpr uint32_t indices_page_size = get_compile_time_arg_val(7);
constexpr uint32_t indices_aligned_page_size = get_compile_time_arg_val(8);

constexpr auto weight_accessor_args = TensorAccessorArgs<9>();
constexpr auto output_accessor_args = TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();
constexpr auto dispatch_table_accessor_args =
    TensorAccessorArgs<output_accessor_args.next_compile_time_args_offset()>();
constexpr auto indices_accessor_args =
    TensorAccessorArgs<dispatch_table_accessor_args.next_compile_time_args_offset()>();
constexpr bool use_dispatch_table_skip =
    get_compile_time_arg_val(indices_accessor_args.next_compile_time_args_offset()) != 0;

constexpr uint32_t TOKENS_PER_CORE = 32;

void kernel_main() {
    // Runtime arg layout: weight_addr, output_addr,
    //   (if dispatch_table_skip: dispatch_table_addr, indices_addr),
    //   token_start_idx.
    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t dispatch_table_addr;
    uint32_t indices_addr;
    uint32_t token_start_idx;
    if constexpr (use_dispatch_table_skip) {
        dispatch_table_addr = get_arg_val<uint32_t>(2);
        indices_addr = get_arg_val<uint32_t>(3);
        token_start_idx = get_arg_val<uint32_t>(4);
    } else {
        dispatch_table_addr = 0;
        indices_addr = 0;
        token_start_idx = get_arg_val<uint32_t>(2);
    }

    constexpr uint32_t weight_tile_size = get_tile_size(cb_weights);
    constexpr uint32_t output_tile_size = get_tile_size(cb_output);

    const auto weight_addrg = TensorAccessor(weight_accessor_args, weight_addr);
    const auto output_addrg = TensorAccessor(output_accessor_args, output_addr);

    // Set to the dispatch_table CB L1 address when skipping — used as a scratch
    // pointer for the per-token has_local test below.
    uint32_t dispatch_table_write_addr = 0;
    uint32_t indices_write_addr = 0;

    if constexpr (use_dispatch_table_skip) {
        const auto dispatch_table_addrg =
            TensorAccessor(dispatch_table_accessor_args, dispatch_table_addr, dispatch_table_page_size);
        const auto indices_addrg = TensorAccessor(indices_accessor_args, indices_addr, indices_page_size);

        // Pre-load dispatch table into CB (c_2) — read once, used by compute
        cb_reserve_back(cb_dispatch_table, dispatch_table_num_pages);
        dispatch_table_write_addr = get_write_ptr(cb_dispatch_table);
        for (uint32_t i = 0; i < dispatch_table_num_pages; i++) {
            noc_async_read_page(
                i, dispatch_table_addrg, dispatch_table_write_addr + i * dispatch_table_aligned_page_size);
        }
        noc_async_read_barrier();
        cb_push_back(cb_dispatch_table, dispatch_table_num_pages);

        // Pre-load indices for this core's tokens into CB (c_3) — read once, used by compute
        cb_reserve_back(cb_indices, indices_pages_per_core);
        indices_write_addr = get_write_ptr(cb_indices);
        for (uint32_t i = 0; i < indices_pages_per_core; i++) {
            uint32_t page_idx = token_start_idx + i;
            noc_async_read_page(page_idx, indices_addrg, indices_write_addr + i * indices_aligned_page_size);
        }
        noc_async_read_barrier();
        cb_push_back(cb_indices, indices_pages_per_core);
    }

    // Phase 1: Stream one weight per expert per token (matching expert-by-expert compute).
    for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CORE; ++token_idx) {
        uint32_t global_token_idx = token_start_idx + token_idx;

        bool has_local = true;
        if constexpr (use_dispatch_table_skip) {
            // Access dispatch table and indices from L1 (data still valid after cb_push_back)
            int32_t* dispatch_table = (int32_t*)dispatch_table_write_addr;
            int32_t* token_indices = (int32_t*)(indices_write_addr + token_idx * indices_aligned_page_size);
            has_local = false;
            for (uint32_t k = 0; k < num_experts; ++k) {
                int32_t expert_id = token_indices[k];
                if (dispatch_table[expert_id] != -1) {
                    has_local = true;
                    break;
                }
            }
        }

        for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            cb_reserve_back(cb_weights, 1);
            uint32_t cb_write_addr = get_write_ptr(cb_weights);

            if constexpr (use_dispatch_table_skip) {
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
            } else {
                uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
                noc_async_read_page(weight_page_idx, weight_addrg, cb_write_addr);
                noc_async_read_barrier();
            }
            cb_push_back(cb_weights, 1);
        }
    }

    // Phase 2: Write output tiles after compute finishes.
    // The output CB holds TOKENS_PER_CORE * emb_dim_cb_tiles tile-sized pages
    // (including padding when emb_dim is not 1024-aligned); only the first
    // emb_dim_out_tiles of them hold real data for this 32-token block.
    constexpr uint32_t cb_output_tiles = emb_dim_cb_tiles * TOKENS_PER_CORE;
    cb_wait_front(cb_output, cb_output_tiles);

    uint32_t cb_read_addr = get_read_ptr(cb_output);

    uint32_t tile_row = token_start_idx / TOKENS_PER_CORE;
    uint32_t start_tile_idx = tile_row * emb_dim_out_tiles;

    for (uint32_t tile_idx = 0; tile_idx < emb_dim_out_tiles; ++tile_idx) {
        noc_async_write_page(start_tile_idx + tile_idx, output_addrg, cb_read_addr);
        cb_read_addr += output_tile_size;
    }

    noc_async_write_barrier();

    cb_pop_front(cb_output, cb_output_tiles);
}
