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
// Raw byte count of this core's column group of one row (the whole row unless the factory split the columns) —
// used for the NoC read so we transfer exactly those bytes and tolerate non-1024-aligned widths.
constexpr uint32_t emb_dim_bytes = get_compile_time_arg_val(2);
// use_local_mask (dispatch-table skip path): the writer pushes, per 32-token chunk, one byte per (token, expert slot)
// into cb_local_mask: 1 = local slot (read its combine row), 2 = the forced accumulator-init slot of a token with no
// local expert (fed zeros: the compute multiplies it by a zero weight, and 0 * an unwritten row could be NaN),
// 0 = non-local slot (the compute skips it; nothing is read). Combine then need not zero-fill its output.
constexpr bool use_local_mask = get_compile_time_arg_val(3) != 0;
constexpr auto combine_accessor_args = TensorAccessorArgs<4>();
constexpr uint32_t cb_local_mask = tt::CBIndex::c_4;

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_chunks = get_arg_val<uint32_t>(2);
    const uint32_t col_off_bytes = get_arg_val<uint32_t>(3);  // this core's column group within a row

    const auto combine_addrg = TensorAccessor(combine_accessor_args, combine_addr);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        volatile tt_l1_ptr uint8_t* mask = nullptr;
        if constexpr (use_local_mask) {
            cb_wait_front(cb_local_mask, 1);
            mask = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_read_ptr(cb_local_mask));
        }
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            // one batch per token: issue every slot's read, then one barrier (the compute consumes slot by slot)
            cb_reserve_back(cb_combine_input, num_experts * emb_dim_cb_tiles);
            const uint32_t base = get_write_ptr(cb_combine_input);
            constexpr uint32_t slot_bytes = emb_dim_cb_tiles * get_tile_size(cb_combine_input);
            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                const uint32_t cb_write_addr = base + expert_idx * slot_bytes;
                const uint32_t m = use_local_mask ? mask[token_idx * num_experts + expert_idx] : 1;
                if (m == 1) {
                    uint32_t expert_page_idx = global_token_idx * num_experts + expert_idx;
                    uint64_t noc_addr = combine_addrg.get_noc_addr(expert_page_idx) + col_off_bytes;
                    noc_async_read(noc_addr, cb_write_addr, emb_dim_bytes);
                } else if (m == 2) {
                    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
                    for (uint32_t off = 0; off < emb_dim_bytes; off += MEM_ZEROS_SIZE) {
                        const uint32_t n = emb_dim_bytes - off < MEM_ZEROS_SIZE ? emb_dim_bytes - off : MEM_ZEROS_SIZE;
                        noc_async_read(zeros, cb_write_addr + off, n);
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_combine_input, num_experts * emb_dim_cb_tiles);
        }
        if constexpr (use_local_mask) {
            cb_pop_front(cb_local_mask, 1);
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
