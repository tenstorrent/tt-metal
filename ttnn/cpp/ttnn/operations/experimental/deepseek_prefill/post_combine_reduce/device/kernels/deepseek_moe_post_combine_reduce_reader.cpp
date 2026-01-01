// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;

constexpr uint32_t num_experts = get_compile_time_arg_val(0);
// Number of tile-sized CB pages needed to hold one row of emb_dim elements
// (ceil(emb_dim / 1024) when emb_dim is a multiple of 32 but not 1024).
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
// Raw byte count of one emb_dim row — used for the NoC read so we transfer
// exactly emb_dim bytes and tolerate non-1024-aligned embedding dims.
constexpr uint32_t emb_dim_bytes = get_compile_time_arg_val(2);
constexpr auto combine_accessor_args = TensorAccessorArgs<3>();
// Row-indexed combine: combine_output is a [rows, emb] table and slot j of token t reads row row_index[t, j]; a
// 32-token chunk of index pages is read at once into cb_row_index. Slots pointing at skip_row are not read: the
// expert skip drops them, so their c_0 slot only has to hold finite data (zeroed once, then earlier real rows).
constexpr uint32_t ri_off = combine_accessor_args.next_compile_time_args_offset();
constexpr bool row_indexed = get_compile_time_arg_val(ri_off) != 0;
constexpr uint32_t skip_row = get_compile_time_arg_val(ri_off + 1);
constexpr uint32_t row_index_aligned_page_size = get_compile_time_arg_val(ri_off + 2);
// The table's row count: a row index at or past it is not read (a bad index must not read past the table).
constexpr uint32_t table_rows = get_compile_time_arg_val(ri_off + 3);
constexpr auto row_index_accessor_args = TensorAccessorArgs<ri_off + 4>();
constexpr uint32_t cb_row_index_id = tt::CBIndex::c_5;

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t combine_addr = get_arg_val<uint32_t>(0);
    uint32_t token_start_idx = get_arg_val<uint32_t>(1);
    uint32_t num_chunks = get_arg_val<uint32_t>(2);

    Noc noc;
    CircularBuffer cb_combine_input(cb_combine_input_id);

    const auto combine_addrg = TensorAccessor(combine_accessor_args, combine_addr);

    uint32_t row_index_l1 = 0;
    if constexpr (row_indexed) {
        CircularBuffer cb_row_index(cb_row_index_id);
        row_index_l1 = cb_row_index.get_write_ptr();
        // c_0 is a single slot: zero it once so a skipped slot never hands compute uninitialized bytes.
        cb_combine_input.reserve_back(emb_dim_cb_tiles);
        noc.async_write_zeros(cb_combine_input, emb_dim_cb_tiles * cb_combine_input.get_tile_size());
        noc.write_zeros_l1_barrier();
    }

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        if constexpr (row_indexed) {
            const auto row_index_addrg =
                TensorAccessor(row_index_accessor_args, get_arg_val<uint32_t>(3), row_index_aligned_page_size);
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
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                cb_combine_input.reserve_back(emb_dim_cb_tiles);

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
                        {.page_id = expert_page_idx},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                }
                cb_combine_input.push_back(emb_dim_cb_tiles);
            }
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
