// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/experimental/paged_cache/device/kernels/dataflow/update_cache_rows.hpp"

void kernel_main() {
    Noc noc;

    const auto my_batch_idx = get_arg(args::my_batch_idx);
    const bool wait_to_start_signal = get_arg(args::wait_to_start) == 1;

    constexpr auto cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto index_stick_size_B = get_arg(args::index_stick_size_B);

    // paged_cache args
    constexpr auto num_heads = get_arg(args::num_heads);
    constexpr auto page_table_stick_size = get_arg(args::page_table_stick_size);

    constexpr auto St = get_arg(args::St);
    constexpr auto input_is_row_major = get_arg(args::input_is_row_major);
    // Rows written per user. Also the fixed number of read-modify-write group slots: one per
    // distinct target tile-row, padded with unused slots so compute's loop count is static.
    constexpr auto num_rows = get_arg(args::num_rows);

    constexpr uint32_t head_offset_t = Wt * St;

    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_input(dfb::input);
#ifdef USE_INDEX_TENSOR
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    // Kick off compute. Tiled input is Wt tiles; row-major is one shard page.
    constexpr uint32_t input_cb_pages = input_is_row_major ? 1 : Wt;
    dfb_input.reserve_back(input_cb_pages);
    dfb_input.push_back(input_cb_pages);

    const uint32_t cache_tile_bytes = dfb_cache.get_tile_size();

    const auto s0 = TensorAccessor(tensor::cache);

#ifdef USE_INDEX_TENSOR
    volatile tt_l1_ptr uint32_t* index_ptr;
    {
        const auto addrg = TensorAccessor(tensor::index);

        dfb_index.reserve_back(1);
        uint32_t index_wr_ptr = dfb_index.get_write_ptr();
        // index_tensor has one page to read
#ifdef ARCH_QUASAR
        // On Quasar DM, get_write_ptr() returns the uncached L1 alias and the NoC won't take it as a
        // destination — pass the DFB endpoint so the NoC resolves the cached address (same write ptr).
        // The RISC reads the landed value back through the uncached alias (index_ptr) below.
        noc.async_read(addrg, dfb_index, index_stick_size_B, {.page_id = 0}, {});
#else
        noc.async_read(addrg, CoreLocalMem<uint32_t>(index_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
#endif
        noc.async_read_barrier();
        dfb_index.push_back(1);
        index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_wr_ptr);
    }
    volatile tt_l1_ptr uint32_t* user_positions = index_ptr + my_batch_idx * num_rows;
    auto position = [user_positions](uint32_t s) -> uint32_t { return user_positions[s]; };
#else
    auto position = [](uint32_t s) -> uint32_t { return get_vararg(s); };
#endif

    volatile tt_l1_ptr uint32_t* page_table_ptr = nullptr;
#ifdef IS_PAGED_CACHE
    {
        const auto page_table_gen = TensorAccessor(tensor::page_table);
        dfb_page_table.reserve_back(1);
        uint32_t page_table_wr_ptr = dfb_page_table.get_write_ptr();
        uint32_t page_table_row = my_batch_idx;
#ifdef PAGE_TABLE_ROW_TENSOR
        {
            // The row lands in the page-table buffer, which the row read below then overwrites.
            constexpr auto page_table_row_page_bytes = get_arg(args::page_table_row_page_bytes);
            const auto row_gen = TensorAccessor(tensor::page_table_row);
#ifdef ARCH_QUASAR
            noc.async_read(row_gen, dfb_page_table, page_table_row_page_bytes, {.page_id = 0}, {});
#else
            noc.async_read(
                row_gen, CoreLocalMem<uint32_t>(page_table_wr_ptr), page_table_row_page_bytes, {.page_id = 0}, {});
#endif
            noc.async_read_barrier();
            page_table_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_wr_ptr)[0];
        }
#endif
#ifdef ARCH_QUASAR
        // See the index read: on Quasar the NoC needs the DFB's cached address.
        noc.async_read(page_table_gen, dfb_page_table, page_table_stick_size, {.page_id = page_table_row}, {});
#else
        noc.async_read(
            page_table_gen,
            CoreLocalMem<uint32_t>(page_table_wr_ptr),
            page_table_stick_size,
            {.page_id = page_table_row},
            {});
#endif
        noc.async_read_barrier();
        dfb_page_table.push_back(1);
        page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_wr_ptr);
    }
#endif

    const UpdateCacheRows<decltype(position)> rows{
        .position = position,
        .batch_tile_offset = my_batch_idx * cache_batch_num_tiles,
        .page_table = page_table_ptr,
    };

    if (wait_to_start_signal) {
        // wait for signal from writer that it has finished using the input buffer
        Semaphore receiver_sem(sem::in0_seq);
        receiver_sem.wait(1);
        receiver_sem.set(0);
    }

    // Distinct groups are distinct tile-rows, so reading the next group ahead of the writer's
    // write-back of the previous one cannot lose an update.
    uint32_t num_groups = 0;
    for (uint32_t s = 0; s < num_rows; ++s) {
        uint32_t cache_id = rows.group_tile_id(s);
        if (cache_id == UPDATE_CACHE_SKIP_ROW) {
            continue;
        }
        ++num_groups;
        for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
            dfb_cache.reserve_back(Wt);
#ifdef ARCH_QUASAR
            // Quasar: DFB endpoint (cached address) + per-tile offset into the Wt-entry reservation,
            // instead of walking the uncached get_write_ptr() alias (see the index read above).
            for (uint32_t t = 0; t < Wt; ++t) {
                noc.async_read(
                    s0, dfb_cache, cache_tile_bytes, {.page_id = cache_id + t}, {.offset_bytes = t * cache_tile_bytes});
            }
#else
            uint32_t cache_l1_write_addr = dfb_cache.get_write_ptr();
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc.async_read(
                    s0, CoreLocalMem<uint32_t>(cache_l1_write_addr), cache_tile_bytes, {.page_id = curr_cache_id}, {});
                cache_l1_write_addr += cache_tile_bytes;
            }
#endif
            noc.async_read_barrier();
            dfb_cache.push_back(Wt);

            cache_id += head_offset_t;
        }
    }

    // Unused group slots: compute still runs, the writer drops the result.
    for (; num_groups < num_rows; ++num_groups) {
        for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
            dfb_cache.reserve_back(Wt);
            dfb_cache.push_back(Wt);
        }
    }
}
