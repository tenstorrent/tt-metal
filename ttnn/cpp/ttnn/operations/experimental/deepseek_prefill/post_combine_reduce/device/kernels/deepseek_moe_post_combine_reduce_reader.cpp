// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;

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
// residual (optional): the chunk's out_tiles tiles of this core's column group -> cb_residual
constexpr uint32_t res_base = combine_accessor_args.next_compile_time_args_offset();
constexpr bool has_residual = get_compile_time_arg_val(res_base) != 0;
constexpr uint32_t out_tiles = get_compile_time_arg_val(res_base + 1);
constexpr uint32_t row_tiles = get_compile_time_arg_val(res_base + 2);
constexpr auto residual_accessor_args = TensorAccessorArgs<res_base + 3>();
constexpr uint32_t cb_residual_id = tt::CBIndex::c_5;
// Row-indexed combine: combine_output is a [rows, emb] table and slot j of token t reads row row_index[t, j]; a
// 32-token chunk of index pages is read at once into cb_row_index. Slots pointing at skip_row are not read: the
// expert skip drops them, so their c_0 slot only has to hold finite data (zeroed once, then earlier real rows).
constexpr uint32_t ri_off = residual_accessor_args.next_compile_time_args_offset();
constexpr bool row_indexed = get_compile_time_arg_val(ri_off) != 0;
constexpr uint32_t skip_row = get_compile_time_arg_val(ri_off + 1);
constexpr uint32_t row_index_aligned_page_size = get_compile_time_arg_val(ri_off + 2);
// The table's row count: a row index at or past it is not read (a bad index must not read past the table).
constexpr uint32_t table_rows = get_compile_time_arg_val(ri_off + 3);
constexpr auto row_index_accessor_args = TensorAccessorArgs<ri_off + 4>();
constexpr uint32_t cb_row_index_id = tt::CBIndex::c_7;
constexpr uint32_t cb_local_mask_id = tt::CBIndex::c_4;

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_chunks = get_arg_val<uint32_t>(2);
    const uint32_t col_off_bytes = get_arg_val<uint32_t>(3);  // this core's column group within a row
    [[maybe_unused]] const uint32_t residual_addr = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t col_tile_start = get_arg_val<uint32_t>(5);

    Noc noc;
    CircularBuffer cb_combine_input(cb_combine_input_id);
    CircularBuffer cb_residual(cb_residual_id);
    CircularBuffer cb_local_mask(cb_local_mask_id);

    const auto combine_addrg = TensorAccessor(combine_accessor_args, combine_addr);
    const uint32_t slot_bytes = emb_dim_cb_tiles * cb_combine_input.get_tile_size();

    uint32_t row_index_l1 = 0;
    if constexpr (row_indexed) {
        CircularBuffer cb_row_index(cb_row_index_id);
        row_index_l1 = cb_row_index.get_write_ptr();
        // Zero all of c_0 (both token batches) once so a skipped slot never hands compute uninitialized bytes.
        constexpr uint32_t c0_tiles = 2 * num_experts * emb_dim_cb_tiles;
        cb_combine_input.reserve_back(c0_tiles);
        noc.async_write_zeros(cb_combine_input, c0_tiles * cb_combine_input.get_tile_size());
        noc.write_zeros_l1_barrier();
    }

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        if constexpr (has_residual) {
            const uint32_t residual_tile_size = cb_residual.get_tile_size();
            const auto res = TensorAccessor(residual_accessor_args, residual_addr, residual_tile_size);
            cb_residual.reserve_back(out_tiles);
            const uint32_t first = (token_start_idx / TOKENS_PER_CHUNK) * row_tiles + col_tile_start;
            for (uint32_t j = 0; j < out_tiles; ++j) {
                noc.async_read(
                    res,
                    cb_residual,
                    residual_tile_size,
                    {.page_id = first + j},
                    {.offset_bytes = j * residual_tile_size});
            }
            noc.async_read_barrier();
            cb_residual.push_back(out_tiles);
        }
        if constexpr (row_indexed) {
            const auto row_index_addrg =
                TensorAccessor(row_index_accessor_args, get_arg_val<uint32_t>(6), row_index_aligned_page_size);
            for (uint32_t i = 0; i < TOKENS_PER_CHUNK; ++i) {
                noc.async_read(
                    row_index_addrg,
                    CoreLocalMem<uint32_t>(row_index_l1 + i * row_index_aligned_page_size),
                    row_index_aligned_page_size,
                    {.page_id = token_start_idx + i},
                    {});
            }
            noc.async_read_barrier();
        }
        volatile tt_l1_ptr uint8_t* mask = nullptr;
        if constexpr (use_local_mask) {
            cb_local_mask.wait_front(1);
            mask = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cb_local_mask.get_read_ptr());
        }
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            // one batch per token: issue every slot's read, then one barrier (the compute consumes slot by slot)
            cb_combine_input.reserve_back(num_experts * emb_dim_cb_tiles);
            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                const uint32_t slot_off = expert_idx * slot_bytes;
                const uint32_t m = use_local_mask ? mask[token_idx * num_experts + expert_idx] : 1;
                if (m == 1) {
                    uint32_t expert_page_idx = global_token_idx * num_experts + expert_idx;
                    if constexpr (row_indexed) {
                        expert_page_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            row_index_l1 + token_idx * row_index_aligned_page_size)[expert_idx];
                        ASSERT(expert_page_idx == skip_row || expert_page_idx < table_rows);
                    }
                    if (!row_indexed || (expert_page_idx != skip_row && expert_page_idx < table_rows)) {
                        noc.async_read(
                            combine_addrg,
                            cb_combine_input,
                            emb_dim_bytes,
                            {.page_id = expert_page_idx, .offset_bytes = col_off_bytes},
                            {.offset_bytes = slot_off});
                    }
                } else if (m == 2) {
                    const uint32_t cb_write_addr = cb_combine_input.get_write_ptr() + slot_off;
                    for (uint32_t off = 0; off < emb_dim_bytes; off += MEM_ZEROS_SIZE) {
                        const uint32_t n = emb_dim_bytes - off < MEM_ZEROS_SIZE ? emb_dim_bytes - off : MEM_ZEROS_SIZE;
                        noc.async_read(
                            UnicastEndpoint{},
                            CoreLocalMem<uint32_t>(cb_write_addr + off),
                            n,
                            {.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = MEM_ZEROS_BASE},
                            {});
                    }
                }
            }
            noc.async_read_barrier();
            cb_combine_input.push_back(num_experts * emb_dim_cb_tiles);
        }
        if constexpr (use_local_mask) {
            cb_local_mask.pop_front(1);
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
