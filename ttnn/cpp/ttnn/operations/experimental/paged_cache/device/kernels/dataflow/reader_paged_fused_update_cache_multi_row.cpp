// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-row variant of reader_paged_fused_update_cache_interleaved_start_id.cpp (rows_per_user > 1).
//
// Each core serves ONE user and writes that user's rows_per_user (T) rows: input tile j (rows
// [j*32, (j+1)*32) of the shard) carries row j, head h on tile row h, and lands at cache position
// index[my_batch_idx * T + j] (index == -1 skips that row). Rows are grouped by the physical cache
// tile they land in; each distinct tile is one read-modify-write pass, so a row group that straddles
// a 32-row tile (or a page-table block) takes two passes and one that does not takes one. The pass
// count is fixed at compile time (num_passes) because compute cannot see the positions: a pass with
// no rows is pushed without reading DRAM, and the writer pops it without writing, exactly the
// skip_update path of the single-row kernels. The grouping below is duplicated verbatim in the
// writer; the two must agree.
//
// A row whose position is past the user's page table (virtual block >= max_blocks_per_seq), or past
// the cache rows of a non-paged cache, is skipped like an index of -1. Looking its block up would read
// beyond this user's page-table stick, and the garbage block id would address an arbitrary DRAM page.
// The verify never stages such a position; the check turns a bad one into a dropped row, not a stray
// write. Valid rows are unaffected.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#ifdef INDEX_IS_DRAM
constexpr bool index_is_dram = true;
#else
constexpr bool index_is_dram = false;
#endif
#ifdef PAGE_TABLE_IS_DRAM
constexpr bool page_table_is_dram = true;
#else
constexpr bool page_table_is_dram = false;
#endif

void kernel_main() {
    Noc noc;

    const auto has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }

    const auto my_batch_idx = get_arg(args::my_batch_idx);

    constexpr auto cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto index_stick_size_B = get_arg(args::index_stick_size_B);

    // paged_cache args
    constexpr auto num_heads = get_arg(args::num_heads);
    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto block_size_t = get_arg(args::block_size_t);
    constexpr auto max_blocks_per_seq = get_arg(args::max_blocks_per_seq);
    constexpr auto page_table_stick_size = get_arg(args::page_table_stick_size);

    constexpr auto St = get_arg(args::St);
    constexpr auto batch_size = get_arg(args::batch_size);
    constexpr auto rows_per_user = get_arg(args::rows_per_user);
    constexpr auto num_passes = get_arg(args::num_passes);

    constexpr uint32_t head_offset_t = Wt * St;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t NO_PASS = 0xFFFFFFFFu;

    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_index(dfb::index);
#ifdef IS_PAGED_CACHE
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    // Kick off compute: the whole T-tile-row shard is resident.
    dfb_input.reserve_back(rows_per_user * Wt);
    dfb_input.push_back(rows_per_user * Wt);

    const uint32_t cache_tile_bytes = dfb_cache.get_entry_size();
    const auto s0 = TensorAccessor(tensor::cache);

    // ---- positions ----
    dfb_index.reserve_back(1);
    const uint32_t index_wr_ptr = dfb_index.get_write_ptr();
#ifdef INDEX_IS_DRAM
    {
        const auto addrg = TensorAccessor(tensor::index);
        noc.async_read(addrg, CoreLocalMem<uint32_t>(index_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
        noc.async_read_barrier();
    }
#endif
    dfb_index.push_back(1);
    volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_wr_ptr);

    uint32_t row_idx[rows_per_user];
    bool any_row = false;
    for (uint32_t j = 0; j < rows_per_user; ++j) {
        row_idx[j] = index_ptr[my_batch_idx * rows_per_user + j];
        any_row |= row_idx[j] != (uint32_t)-1;
    }

    // ---- row -> physical cache tile -> pass ----
    uint32_t pass_cache_id[num_passes];
    uint32_t num_active_passes = 0;
    if (any_row) {
#ifdef IS_PAGED_CACHE
        uint32_t num_pages_to_read = page_table_is_dram ? 1 : batch_size;
        dfb_page_table.reserve_back(num_pages_to_read);
        uint32_t page_table_wr_ptr = dfb_page_table.get_write_ptr();
#ifdef PAGE_TABLE_IS_DRAM
        {
            const auto page_table_gen = TensorAccessor(tensor::page_table);
            noc.async_read(
                page_table_gen,
                CoreLocalMem<uint32_t>(page_table_wr_ptr),
                page_table_stick_size,
                {.page_id = my_batch_idx},
                {});
            noc.async_read_barrier();
        }
#else
        page_table_wr_ptr += my_batch_idx * page_table_stick_size;
#endif
        dfb_page_table.push_back(num_pages_to_read);
        // DRAM uses uint32 entries; a page table sharded into SRAM uses uint16 entries
        volatile tt_l1_ptr uint32_t* page_table_ptr_u32 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_wr_ptr);
        volatile tt_l1_ptr uint16_t* page_table_ptr_u16 =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_wr_ptr);
#endif
        for (uint32_t j = 0; j < rows_per_user; ++j) {
            const uint32_t update_idx = row_idx[j];
            if (update_idx == (uint32_t)-1) {
                continue;
            }
#ifdef IS_PAGED_CACHE
            const uint32_t virtual_block_id = update_idx / block_size;
            if (virtual_block_id >= max_blocks_per_seq) {
                continue;  // past this user's page-table stick: skipped (see the header), never an unchecked write
            }
            const uint32_t physical_block_id = (page_table_is_dram)
                                                   ? page_table_ptr_u32[virtual_block_id]
                                                   : static_cast<uint32_t>(page_table_ptr_u16[virtual_block_id]);
            const uint32_t cache_id = physical_block_id * num_heads * block_size_t * Wt +
                                      ((update_idx % block_size) / TILE_HEIGHT) * Wt;
#else
            if (update_idx >= St * TILE_HEIGHT) {
                continue;  // past this user's cache rows: skipped, as above
            }
            const uint32_t cache_id = my_batch_idx * cache_batch_num_tiles + (update_idx / TILE_HEIGHT) * Wt;
#endif
            uint32_t p = 0;
            while (p < num_active_passes && pass_cache_id[p] != cache_id) {
                ++p;
            }
            if (p == num_active_passes) {
                pass_cache_id[num_active_passes++] = cache_id;
            }
        }
    }

    // ---- one read per (pass, head); empty passes push without reading ----
    for (uint32_t p = 0; p < num_passes; ++p) {
        const bool active = p < num_active_passes;
        uint32_t cache_id = active ? pass_cache_id[p] : NO_PASS;
        for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
            dfb_cache.reserve_back(Wt);
            if (active) {
                uint32_t cache_l1_write_addr = dfb_cache.get_write_ptr();
                for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc.async_read(
                        s0,
                        CoreLocalMem<uint32_t>(cache_l1_write_addr),
                        cache_tile_bytes,
                        {.page_id = curr_cache_id},
                        {});
                    cache_l1_write_addr += cache_tile_bytes;
                }
                noc.async_read_barrier();
                cache_id += head_offset_t;
            }
            dfb_cache.push_back(Wt);
        }
    }
}
