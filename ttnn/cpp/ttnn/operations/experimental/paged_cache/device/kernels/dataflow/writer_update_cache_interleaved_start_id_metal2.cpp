// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    const uint32_t cache_start_id = get_arg(args::cache_start_id);
    uint32_t cache_tile_offset_B = get_arg(args::cache_tile_offset_B);
    const uint32_t my_batch_idx = get_arg(args::my_batch_idx);
    const bool send_signal = get_arg(args::send_signal) == 1;
    const uint32_t send_core_x = get_arg(args::send_core_x);
    const uint32_t send_core_y = get_arg(args::send_core_y);

    constexpr uint32_t cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Wbytes = get_arg(args::Wbytes);

    // paged_cache args
    constexpr uint32_t num_heads = get_arg(args::num_heads);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t block_size_t = get_arg(args::block_size_t);
    constexpr uint32_t max_blocks_per_seq = get_arg(args::max_blocks_per_seq);

    constexpr uint32_t St = get_arg(args::St);
    // 0 = legacy unbounded behavior; nonzero = wrap update_idx mod this value before
    // page_table lookup (bounded sliding-window cache support).
    constexpr uint32_t cache_position_modulo = get_arg(args::cache_position_modulo);

    constexpr uint32_t head_offset_t = Wt * St;

    constexpr uint32_t TILE_HEIGHT = 32;

    const auto s0 = TensorAccessor(tensor::cache);

    // NOTE: this kernel's `cache` DFB is the *output* buffer (the host binds its OUTPUT
    // DataflowBufferSpec here) — it holds the re-tilized cache block this kernel writes back to the
    // cache tensor, not the cache tiles the reader pulled in. The `untilized_input` DFB likewise
    // carries the untilized *input* shard that compute produced.
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_untilized_cache(dfb::untilized_cache);
    DataflowBuffer dfb_untilized_cache2(dfb::untilized_cache2);
    DataflowBuffer dfb_untilized_input(dfb::untilized_input);
#ifdef USE_INDEX_TENSOR
    // Bound only in index-tensor mode; the reader fills it and this kernel reads the value back out.
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    // Bound only for a paged cache; likewise reader-filled, read here.
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    const uint32_t cache_tile_bytes = dfb_cache.get_tile_size();

    uint32_t cache_id = cache_start_id;
    uint32_t update_idx = 0;

    bool skip_update = false;

#ifdef USE_INDEX_TENSOR
    {
        dfb_index.wait_front(1);
        uint32_t index_dfb_ptr = dfb_index.get_read_ptr();
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_dfb_ptr);
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
            dfb_page_table.wait_front(1);
            uint32_t page_table_dfb_rd_ptr = dfb_page_table.get_read_ptr();
            volatile tt_l1_ptr uint32_t* page_table_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_dfb_rd_ptr);

            const uint32_t virtual_block_id = update_idx / block_size;
            const uint32_t physical_block_id = page_table_ptr[virtual_block_id];
            const uint32_t block_start_id = physical_block_id * num_heads * block_size_t * Wt;
            const uint32_t block_row_tile = (update_idx % block_size) / TILE_HEIGHT;
            const uint32_t block_offset = block_row_tile * Wt;
            cache_id = block_start_id + block_offset;

            // Page-table value consumed; pop to balance the wait above.
            dfb_page_table.pop_front(1);
#else
            const uint32_t cache_batch_tile_offset = my_batch_idx * cache_batch_num_tiles;
            const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
            cache_id = cache_start_id;
#endif
            cache_tile_offset_B = update_idx % TILE_HEIGHT * Wbytes;
        }
        // The index value is consumed on both the skip and update paths; the reader pushes
        // dfb_index unconditionally, so pop it here (outside the skip branch) to balance the wait.
        dfb_index.pop_front(1);
    }
#endif

    dfb_untilized_input.wait_front(Wt);  // input tensor
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    uint32_t input_l1_read_addr = dfb_untilized_input.get_read_ptr();
    UnicastEndpoint local_src;

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Wait on compute to untilize a block. Update that block in L1.
        dfb_untilized_cache.wait_front(Wt);
        dfb_untilized_cache2.reserve_back(Wt);

        // dfb_untilized_cache and dfb_untilized_cache2 are aliased: they name the same L1 region
        // through two logical buffers. The new row is written in place through the first, then
        // republished through the second for compute to re-tilize.
        uint32_t cache_l1_write_addr = dfb_untilized_cache.get_read_ptr() + cache_tile_offset_B;
        noc.async_read(
            local_src,
            CoreLocalMem<uint32_t>(cache_l1_write_addr),
            Wbytes,
            {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = input_l1_read_addr},
            {});
        noc.async_read_barrier();
        dfb_untilized_cache2.push_back(Wt);
        dfb_untilized_cache.pop_front(Wt);  // NEW

        // Wait on compute to tilize an updated block. Write that block to DRAM
        dfb_cache.wait_front(Wt);
        if (!skip_update) {
            uint32_t out_l1_read_addr = dfb_cache.get_read_ptr();
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(out_l1_read_addr), s0, cache_tile_bytes, {}, {.page_id = curr_cache_id});
                out_l1_read_addr += cache_tile_bytes;
            }

            noc.async_writes_flushed();
        }
        dfb_cache.pop_front(Wt);

        if (!skip_update) {
            // Delay syncing the writes to maximize perf.
            noc.async_write_barrier();
        }

        // read from next head
        input_l1_read_addr += Wbytes;
        cache_id += head_offset_t;
    }

    dfb_untilized_input.pop_front(Wt);

    if (send_signal) {
        // send signal to receiver core that we are done using the input DFB
        Semaphore<>(sem::receiver).up(noc, send_core_x, send_core_y, 1);
        noc.async_atomic_barrier();
    }
}
