// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// Streams only the active (token, slot) pairs to compute. A slot is active when its expert is local to this
// dispatch group (dispatch-table mode) or its routing weight is non-zero (weight mode). Per 32-token chunk the
// reader first publishes each token's active-slot count (c_4), then pushes one combine row (c_0) and one
// weight scalar (element 0 of a c_1 tile) per active slot, in token then slot order.

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;
constexpr uint32_t cb_weights_id = tt::CBIndex::c_1;
constexpr uint32_t cb_dispatch_table_id = tt::CBIndex::c_2;
constexpr uint32_t cb_indices_id = tt::CBIndex::c_3;
constexpr uint32_t cb_token_counts_id = tt::CBIndex::c_4;
constexpr uint32_t cb_weight_scratch_id = tt::CBIndex::c_5;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
// Tile-sized CB pages holding one emb_dim row (ceil(emb_dim / 1024)).
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
// Raw byte count of one emb_dim row (tolerates non-1024-aligned embedding dims).
constexpr uint32_t emb_dim_bytes = get_compile_time_arg_val(2);
constexpr uint32_t weight_aligned_page_size = get_compile_time_arg_val(3);
// Dispatch-table mode only (zero otherwise).
constexpr uint32_t dispatch_table_num_pages = get_compile_time_arg_val(4);
constexpr uint32_t dispatch_table_aligned_page_size = get_compile_time_arg_val(5);
constexpr uint32_t dispatch_table_entries = get_compile_time_arg_val(6);
constexpr uint32_t indices_aligned_page_size = get_compile_time_arg_val(7);
constexpr bool use_dispatch_table_skip = get_compile_time_arg_val(8) != 0;

constexpr auto combine_accessor_args = TensorAccessorArgs<9>();
constexpr auto weight_accessor_args = TensorAccessorArgs<combine_accessor_args.next_compile_time_args_offset()>();
// In weight mode these two carry the weight tensor's args as placeholders; every use is behind
// `if constexpr (use_dispatch_table_skip)`.
constexpr auto dispatch_table_accessor_args =
    TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();
constexpr auto indices_accessor_args =
    TensorAccessorArgs<dispatch_table_accessor_args.next_compile_time_args_offset()>();

constexpr uint32_t TOKENS_PER_CHUNK = 32;
static_assert(num_experts <= 32, "the per-token active-slot mask is a uint32_t");

void kernel_main() {
    // Runtime args: combine_addr, weight_addr, (dispatch-table mode: dispatch_table_addr, indices_addr),
    // token_start_idx, num_chunks.
    uint32_t arg = 0;
    const uint32_t combine_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(arg++);
    uint32_t dispatch_table_addr = 0;
    uint32_t indices_addr = 0;
    if constexpr (use_dispatch_table_skip) {
        dispatch_table_addr = get_arg_val<uint32_t>(arg++);
        indices_addr = get_arg_val<uint32_t>(arg++);
    }
    uint32_t token_start_idx = get_arg_val<uint32_t>(arg++);
    const uint32_t num_chunks = get_arg_val<uint32_t>(arg++);

    Noc noc;
    CircularBuffer cb_combine_input(cb_combine_input_id);
    CircularBuffer cb_weights(cb_weights_id);
    CircularBuffer cb_dispatch_table(cb_dispatch_table_id);
    CircularBuffer cb_indices(cb_indices_id);
    CircularBuffer cb_token_counts(cb_token_counts_id);
    CircularBuffer cb_weight_scratch(cb_weight_scratch_id);

    const auto combine_addrg = TensorAccessor(combine_accessor_args, combine_addr);
    const auto weight_addrg = TensorAccessor(weight_accessor_args, weight_addr);

    // c_2, c_3 and c_5 are reader-private scratch: written and read here, never pushed.
    const uint32_t weight_scratch_l1 = cb_weight_scratch.get_write_ptr();
    const auto weight_at = [&](uint32_t t, uint32_t k) -> uint16_t {
        return *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
            weight_scratch_l1 + (t * num_experts + k) * weight_aligned_page_size);
    };
    const auto read_weight = [&](uint32_t t, uint32_t k) {
        noc.async_read(
            weight_addrg,
            cb_weight_scratch,
            weight_aligned_page_size,
            {.page_id = (token_start_idx + t) * num_experts + k},
            {.offset_bytes = (t * num_experts + k) * weight_aligned_page_size});
    };

    uint32_t dispatch_table_l1 = 0;
    uint32_t indices_l1 = 0;
    if constexpr (use_dispatch_table_skip) {
        const auto dispatch_table_addrg = TensorAccessor(dispatch_table_accessor_args, dispatch_table_addr);
        dispatch_table_l1 = cb_dispatch_table.get_write_ptr();
        indices_l1 = cb_indices.get_write_ptr();
        for (uint32_t i = 0; i < dispatch_table_num_pages; i++) {
            noc.async_read(
                dispatch_table_addrg,
                cb_dispatch_table,
                dispatch_table_aligned_page_size,
                {.page_id = i},
                {.offset_bytes = i * dispatch_table_aligned_page_size});
        }
        noc.async_read_barrier();
    }

    uint32_t active_mask[TOKENS_PER_CHUNK];

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Phase 1: pick the active slots and land their weights in c_5.
        if constexpr (use_dispatch_table_skip) {
            const auto indices_addrg = TensorAccessor(indices_accessor_args, indices_addr);
            for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
                noc.async_read(
                    indices_addrg,
                    cb_indices,
                    indices_aligned_page_size,
                    {.page_id = token_start_idx + t},
                    {.offset_bytes = t * indices_aligned_page_size});
            }
            noc.async_read_barrier();
            auto* dispatch_table = reinterpret_cast<volatile tt_l1_ptr int32_t*>(dispatch_table_l1);
            for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
                auto* token_indices =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(indices_l1 + t * indices_aligned_page_size);
                uint32_t mask = 0;
                for (uint32_t k = 0; k < num_experts; ++k) {
                    const uint32_t expert_id = token_indices[k];
                    // -1 = expert owned by another dispatch group; out-of-range ids are never local.
                    if (expert_id < dispatch_table_entries && dispatch_table[expert_id] != -1) {
                        mask |= 1u << k;
                        read_weight(t, k);
                    }
                }
                active_mask[t] = mask;
            }
            noc.async_read_barrier();
        } else {
            for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
                for (uint32_t k = 0; k < num_experts; ++k) {
                    read_weight(t, k);
                }
            }
            noc.async_read_barrier();
            for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
                uint32_t mask = 0;
                for (uint32_t k = 0; k < num_experts; ++k) {
                    if ((weight_at(t, k) & 0x7FFF) != 0) {  // bf16 +0 and -0 are inactive
                        mask |= 1u << k;
                    }
                }
                active_mask[t] = mask;
            }
        }

        cb_token_counts.reserve_back(1);
        auto* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_token_counts.get_write_ptr());
        for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
            uint32_t n = 0;
            for (uint32_t mask = active_mask[t]; mask != 0; mask &= mask - 1) {
                ++n;
            }
            counts[t] = n;
        }
        cb_token_counts.push_back(1);

        // Phase 2: one combine row + its weight scalar per active slot.
        for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
            const uint32_t global_token_idx = token_start_idx + t;
            for (uint32_t k = 0; k < num_experts; ++k) {
                if ((active_mask[t] & (1u << k)) == 0) {
                    continue;
                }
                cb_combine_input.reserve_back(emb_dim_cb_tiles);
                cb_weights.reserve_back(1);
                noc.async_read(
                    combine_addrg,
                    cb_combine_input,
                    emb_dim_bytes,
                    {.page_id = global_token_idx * num_experts + k},
                    {.offset_bytes = 0});
                // The scalar broadcast reads element 0 of the weight tile.
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_weights.get_write_ptr()) = weight_at(t, k);
                noc.async_read_barrier();
                cb_weights.push_back(1);
                cb_combine_input.push_back(emb_dim_cb_tiles);
            }
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
