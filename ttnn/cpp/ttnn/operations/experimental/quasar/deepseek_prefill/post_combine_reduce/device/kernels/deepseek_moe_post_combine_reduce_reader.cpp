// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"

// L1 address the CPU accesses a scratchpad through: on Quasar DM the uncached alias, as DataflowBuffer pointers are,
// since the CPU caches are not coherent with NoC writes to L1; on WH/BH the plain address.
template <typename T>
uint32_t cpu_l1_addr(const Scratchpad<T>& scratch) {
    uintptr_t addr = local_mem::detail::AddressAccess::address(scratch.local_mem());
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    addr += MEM_L1_UNCACHED_BASE;
#endif
    return static_cast<uint32_t>(addr);
}

constexpr uint32_t num_experts = get_arg(args::num_experts);
// Number of tile-sized DFB entries needed to hold one row of emb_dim elements
// (ceil(emb_dim / 1024) when emb_dim is a multiple of 32 but not 1024).
constexpr uint32_t emb_dim_dfb_tiles = get_arg(args::emb_dim_dfb_tiles);
// Raw byte count of one emb_dim row — used for the NoC read so we transfer
// exactly emb_dim bytes and tolerate non-1024-aligned embedding dims.
constexpr uint32_t emb_dim_bytes = get_arg(args::emb_dim_bytes);
// Experts read per NoC barrier: num_experts (all of a token's rows at once) when the factory sized
// combine_input for it, else 1. One barrier per 8 KB row left the reader latency-bound.
constexpr uint32_t experts_per_batch = get_arg(args::experts_per_batch);
// Row-indexed combine (ROW_INDEXED): combine_output is a [rows, emb] table and slot j of token t reads row
// row_index[t, j]; a 32-token chunk of index pages is read at once into the row_index scratchpad. Slots pointing at
// skip_row are not read: the expert skip drops them, so their combine_input slot only has to hold finite data (zeroed
// once, then earlier real rows).
#ifdef ROW_INDEXED
constexpr bool row_indexed = true;
#else
constexpr bool row_indexed = false;
#endif
constexpr uint32_t skip_row = get_arg(args::skip_row);
constexpr uint32_t row_index_aligned_page_size = get_arg(args::row_index_aligned_page_size);
// The table's row count: a row index at or past it is not read (a bad index must not read past the table).
constexpr uint32_t table_rows = get_arg(args::table_rows);
// The zero-once above covers a single combine_input slot, so the factory does not batch row-indexed reads.
static_assert(!row_indexed || experts_per_batch == 1);

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t token_start_idx = get_arg(args::token_start_idx);
    uint32_t num_chunks = get_arg(args::num_chunks);

    Noc noc;
    DataflowBuffer dfb_combine_input(dfb::combine_input);

    const auto combine_addrg = TensorAccessor(tensor::combine);
    const uint32_t input_tile_size = dfb_combine_input.get_tile_size();

    uint32_t row_index_l1 = 0;
#ifdef ROW_INDEXED
    {
        Scratchpad<uint32_t> row_index_scratch(scratch::row_index);
        row_index_l1 = cpu_l1_addr(row_index_scratch);
        // combine_input is a single entry: zero it once so a skipped entry never hands compute uninitialized bytes.
        dfb_combine_input.reserve_back(emb_dim_dfb_tiles);
        noc.async_write_zeros(dfb_combine_input, emb_dim_dfb_tiles * dfb_combine_input.get_tile_size());
        noc.write_zeros_l1_barrier();
    }
#endif

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
#ifdef ROW_INDEXED
        {
            const auto row_index_addrg = TensorAccessor(tensor::row_index);
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
#endif
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            for (uint32_t expert_base = 0; expert_base < num_experts; expert_base += experts_per_batch) {
                dfb_combine_input.reserve_back(experts_per_batch * emb_dim_dfb_tiles);
                for (uint32_t e = 0; e < experts_per_batch; ++e) {
                    const uint32_t expert_idx = expert_base + e;
                    uint32_t expert_page_idx = global_token_idx * num_experts + expert_idx;
                    if constexpr (row_indexed) {
                        expert_page_idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            row_index_l1 + token_idx * row_index_aligned_page_size)[expert_idx];
                        ASSERT(expert_page_idx == skip_row || expert_page_idx < table_rows);
                    }
                    if (row_indexed && (expert_page_idx == skip_row || expert_page_idx >= table_rows)) {
                        continue;
                    }
                    noc.async_read(
                        combine_addrg,
                        dfb_combine_input,
                        emb_dim_bytes,
                        {.page_id = expert_page_idx},
                        {.offset_bytes = e * emb_dim_dfb_tiles * input_tile_size});
                }
                noc.async_read_barrier();
                dfb_combine_input.push_back(experts_per_batch * emb_dim_dfb_tiles);
            }
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
