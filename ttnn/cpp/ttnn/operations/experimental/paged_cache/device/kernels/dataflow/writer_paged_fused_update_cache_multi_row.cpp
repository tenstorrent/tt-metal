// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-row variant of writer_paged_fused_update_cache_interleaved_start_id.cpp (rows_per_user > 1).
// See reader_paged_fused_update_cache_multi_row.cpp for the row -> pass grouping (and the skip of
// rows past the page table), which this kernel recomputes identically from the same index /
// page-table buffers the reader pushed.
//
// Per (pass, head): wait for compute's untilized cache block, copy in every row of this user that
// lands in the pass's tile (row j, head h comes from untilized input block j, row h, and goes to
// row index[j] % 32), hand the block back for tilize, then write the retilized Wt tiles to DRAM.
// Rows are copied in increasing j, so if two rows name one position the later row wins, which is
// what T sequential single-row updates leave behind.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

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
    constexpr auto Wbytes = get_arg(args::Wbytes);

    // paged_cache args
    constexpr auto num_heads = get_arg(args::num_heads);
    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto block_size_t = get_arg(args::block_size_t);
    constexpr auto max_blocks_per_seq = get_arg(args::max_blocks_per_seq);

    constexpr auto St = get_arg(args::St);
    constexpr uint32_t head_offset_t = Wt * St;
    constexpr auto batch_size = get_arg(args::batch_size);
    constexpr auto page_table_stick_size = get_arg(args::page_table_stick_size);
    constexpr auto rows_per_user = get_arg(args::rows_per_user);
    constexpr auto num_passes = get_arg(args::num_passes);

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t NO_PASS = 0xFFFFFFFFu;
    // Untilized input: block j (= input tile row j) is TILE_HEIGHT rows of Wbytes each.
    constexpr uint32_t input_block_bytes = TILE_HEIGHT * Wbytes;

    const auto s0 = TensorAccessor(tensor::cache);

    // `dfb::cache` is this kernel's name for the *output* buffer (compute's retilized block).
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_untilized_cache(dfb::untilized_cache);
    DataflowBuffer dfb_untilized_cache2(dfb::untilized_cache2);
    DataflowBuffer dfb_untilized_input(dfb::untilized_input);
    DataflowBuffer dfb_index(dfb::index);
#ifdef IS_PAGED_CACHE
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    const uint32_t cache_tile_bytes = dfb_cache.get_entry_size();

    // ---- positions (same derivation as the reader) ----
    dfb_index.wait_front(1);
    volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_index.get_read_ptr());
    uint32_t row_idx[rows_per_user];
    uint32_t row_pass[rows_per_user];
    bool any_row = false;
    for (uint32_t j = 0; j < rows_per_user; ++j) {
        row_idx[j] = index_ptr[my_batch_idx * rows_per_user + j];
        row_pass[j] = NO_PASS;
        any_row |= row_idx[j] != (uint32_t)-1;
    }
    dfb_index.pop_front(1);

    uint32_t pass_cache_id[num_passes];
    uint32_t num_active_passes = 0;
    if (any_row) {
#ifdef IS_PAGED_CACHE
        uint32_t num_pages_to_read = page_table_is_dram ? 1 : batch_size;
        dfb_page_table.wait_front(num_pages_to_read);
        uint32_t page_table_rd_ptr = dfb_page_table.get_read_ptr();
        if constexpr (!page_table_is_dram) {
            page_table_rd_ptr += my_batch_idx * page_table_stick_size;
        }
        // DRAM uses uint32 entries; a page table sharded into SRAM uses uint16 entries
        volatile tt_l1_ptr uint32_t* page_table_ptr_u32 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_rd_ptr);
        volatile tt_l1_ptr uint16_t* page_table_ptr_u16 =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_rd_ptr);
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
            row_pass[j] = p;
        }
#ifdef IS_PAGED_CACHE
        // Page-table pages consumed; pop the same count waited above to balance the buffer.
        dfb_page_table.pop_front(num_pages_to_read);
#endif
    }

    dfb_untilized_input.wait_front(rows_per_user * Wt);  // all T untilized input blocks
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    const uint32_t input_l1_base = dfb_untilized_input.get_read_ptr();
    UnicastEndpoint local_src;

    for (uint32_t p = 0; p < num_passes; ++p) {
        const bool active = p < num_active_passes;
        uint32_t cache_id = active ? pass_cache_id[p] : NO_PASS;
        for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
            // Wait on compute to untilize a block. Update that block in SRAM.
            dfb_untilized_cache.wait_front(Wt);
            dfb_untilized_cache2.reserve_back(Wt);

            const uint32_t cache_block_l1 = dfb_untilized_cache.get_read_ptr();
            for (uint32_t j = 0; j < rows_per_user; ++j) {
                if (row_pass[j] != p) {
                    continue;
                }
                noc.async_read(
                    local_src,
                    CoreLocalMem<uint32_t>(cache_block_l1 + (row_idx[j] % TILE_HEIGHT) * Wbytes),
                    Wbytes,
                    {.noc_x = my_noc_x,
                     .noc_y = my_noc_y,
                     .addr = input_l1_base + j * input_block_bytes + cur_head * Wbytes},
                    {});
                // One row at a time keeps the j order the "later row wins" rule relies on.
                noc.async_read_barrier();
            }
            dfb_untilized_cache2.push_back(Wt);
            dfb_untilized_cache.pop_front(Wt);

            // Wait on compute to tilize the updated block. Write that block to DRAM.
            dfb_cache.wait_front(Wt);
            if (active) {
                uint32_t out_l1_read_addr = dfb_cache.get_read_ptr();
                for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(out_l1_read_addr), s0, cache_tile_bytes, {}, {.page_id = curr_cache_id});
                    out_l1_read_addr += cache_tile_bytes;
                }
                noc.async_writes_flushed();
            }
            dfb_cache.pop_front(Wt);

            if (active) {
                noc.async_write_barrier();
                cache_id += head_offset_t;
            }
        }
    }

    dfb_untilized_input.pop_front(rows_per_user * Wt);
}
