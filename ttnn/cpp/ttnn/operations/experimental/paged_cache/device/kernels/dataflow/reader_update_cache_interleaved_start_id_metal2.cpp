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

void kernel_main() {
    Noc noc;

    const uint32_t cache_start_id = get_arg(args::cache_start_id);
    const uint32_t my_batch_idx = get_arg(args::my_batch_idx);
    const bool wait_to_start_signal = get_arg(args::wait_to_start) == 1;

    constexpr uint32_t cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    constexpr uint32_t Wt = get_arg(args::Wt);
    const uint32_t log_base_2_of_page_size = get_arg(args::log_base_2_of_page_size);
    const uint32_t index_stick_size_B = get_arg(args::index_stick_size_B);

    // paged_cache args
    constexpr uint32_t num_heads = get_arg(args::num_heads);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t block_size_t = get_arg(args::block_size_t);
    constexpr uint32_t max_blocks_per_seq = get_arg(args::max_blocks_per_seq);
    constexpr uint32_t log2_page_table_stick_size = get_arg(args::log2_page_table_stick_size);
    constexpr uint32_t page_table_stick_size = get_arg(args::page_table_stick_size);

    const uint32_t St = get_arg(args::St);
    // 0 = legacy unbounded behavior; nonzero = wrap update_idx mod this value before
    // page_table lookup (bounded sliding-window cache support).
    constexpr uint32_t cache_position_modulo = get_arg(args::cache_position_modulo);

    constexpr uint32_t head_offset_t = Wt * St;

    // dfb::cache holds the cache tiles this reader pulls in from the cache tensor; dfb::input is the
    // resident input shard, which the reader only publishes (its memory is borrowed from the input
    // tensor, so there is nothing to transfer). Both are consumed by compute.
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_input(dfb::input);
#ifdef USE_INDEX_TENSOR
    // Allocated and bound only in index-tensor mode: the reader fills it from the index tensor and
    // the writer reads the same value back out of it.
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    // Allocated and bound only for a paged cache; likewise reader-filled, writer-read.
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    // Kick off compute
    dfb_input.reserve_back(Wt);
    dfb_input.push_back(Wt);

    const uint32_t cache_tile_bytes = dfb_cache.get_tile_size();

    constexpr uint32_t TILE_HEIGHT = 32;

    uint32_t cache_id = cache_start_id;

    const auto s0 = TensorAccessor(tensor::cache);

    bool skip_update = false;

#ifdef USE_INDEX_TENSOR
    {
        const auto addrg = TensorAccessor(tensor::index);

        dfb_index.reserve_back(1);
        uint32_t index_dfb_wr_ptr = dfb_index.get_write_ptr();
        // index_tensor has one page to read
        noc.async_read(addrg, CoreLocalMem<uint32_t>(index_dfb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
        noc.async_read_barrier();
        dfb_index.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_dfb_wr_ptr);

        const uint32_t raw_update_idx = index_ptr[my_batch_idx];
        if (raw_update_idx == (uint32_t)-1) {
            // Passing update_idx = -1 tells us to skip update for this user
            skip_update = true;
        } else {
            // Wrap into the bounded sliding-window cache when enabled, so positions past
            // the physical capacity are addressed correctly (cache_position_modulo is a
            // multiple of block_size, so this preserves the intra-block offset).
            const uint32_t update_idx =
                cache_position_modulo > 0 ? raw_update_idx % cache_position_modulo : raw_update_idx;
#ifdef IS_PAGED_CACHE
            const auto page_table_gen = TensorAccessor(tensor::page_table);
            dfb_page_table.reserve_back(1);
            uint32_t page_table_dfb_wr_ptr = dfb_page_table.get_write_ptr();
            noc.async_read(
                page_table_gen,
                CoreLocalMem<uint32_t>(page_table_dfb_wr_ptr),
                page_table_stick_size,
                {.page_id = my_batch_idx},
                {});
            noc.async_read_barrier();
            dfb_page_table.push_back(1);
            volatile tt_l1_ptr uint32_t* page_table_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_dfb_wr_ptr);

            const uint32_t virtual_block_id = update_idx / block_size;
            const uint32_t physical_block_id = page_table_ptr[virtual_block_id];
            const uint32_t block_start_id = physical_block_id * num_heads * block_size_t * Wt;
            const uint32_t block_row_tile = (update_idx % block_size) / TILE_HEIGHT;
            const uint32_t block_offset = block_row_tile * Wt;
            cache_id = block_start_id + block_offset;
#else
            const uint32_t cache_batch_tile_offset = my_batch_idx * cache_batch_num_tiles;
            const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
            cache_id = cache_start_id;
#endif
        }
    }
#endif

    if (wait_to_start_signal) {
        // wait for signal from writer that it has finished using the input DFB
        Semaphore<> receiver_sem(sem::receiver);
        receiver_sem.wait(1);
        receiver_sem.set(0);
    }

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        dfb_cache.reserve_back(Wt);
        if (!skip_update) {
            uint32_t cache_l1_write_addr = dfb_cache.get_write_ptr();
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc.async_read(
                    s0, CoreLocalMem<uint32_t>(cache_l1_write_addr), cache_tile_bytes, {.page_id = curr_cache_id}, {});
                cache_l1_write_addr += cache_tile_bytes;
            }

            noc.async_read_barrier();
        }
        dfb_cache.push_back(Wt);

        cache_id += head_offset_t;
    }
}
