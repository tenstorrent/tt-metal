// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

// Whether the page table lives in DRAM. When it does not, its buffer is built on the tensor's own
// resident shard and holds one stick per batch entry, addressed by offsetting the read pointer.
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

    const auto cache_start_id = get_arg(args::cache_start_id);
    auto cache_tile_offset_B = get_arg(args::cache_tile_offset_B);
    const auto my_batch_idx = get_arg(args::my_batch_idx);
    const bool send_signal = get_arg(args::send_signal) == 1;
    const auto send_core_x = get_arg(args::send_core_x);
    const auto send_core_y = get_arg(args::send_core_y);

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

    constexpr uint32_t TILE_HEIGHT = 32;

    const auto s0 = TensorAccessor(tensor::cache);

    // `dfb::cache` is this kernel's name for the *output* buffer, which compute fills with the
    // retilized block; the reader's same-named handle is the cache buffer it read from DRAM.
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_untilized_cache(dfb::untilized_cache);
    DataflowBuffer dfb_untilized_cache2(dfb::untilized_cache2);
    DataflowBuffer dfb_untilized_input(dfb::untilized_input);
#ifdef USE_INDEX_TENSOR
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    const uint32_t cache_tile_bytes = dfb_cache.get_tile_size();

    uint32_t cache_id = cache_start_id;
    uint32_t update_idx = 0;

    bool skip_update = false;

#ifdef USE_INDEX_TENSOR
    {
        dfb_index.wait_front(1);
        uint32_t index_rd_ptr = dfb_index.get_read_ptr();
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_rd_ptr);
        const uint32_t update_idx = index_ptr[my_batch_idx];

        if (update_idx == (uint32_t)-1) {
            // Passing update_idx = -1 tells us to skip update for this user
            skip_update = true;
        } else {
#ifdef IS_PAGED_CACHE
            {
                uint32_t num_pages_to_read = page_table_is_dram ? 1 : batch_size;
                dfb_page_table.wait_front(num_pages_to_read);
                uint32_t page_table_rd_ptr = dfb_page_table.get_read_ptr();
                if constexpr (!page_table_is_dram) {
                    page_table_rd_ptr += my_batch_idx * page_table_stick_size;
                }
                // DRAM uses uint32 entries; a page table sharded into SRAM uses uint16 entries
                volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
                volatile tt_l1_ptr uint16_t* page_table_ptr_u16 = nullptr;
                if constexpr (page_table_is_dram) {
                    page_table_ptr_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_rd_ptr);
                } else {
                    page_table_ptr_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_rd_ptr);
                }

                const uint32_t virtual_block_id = update_idx / block_size;
                const uint32_t physical_block_id = (page_table_is_dram)
                                                       ? page_table_ptr_u32[virtual_block_id]
                                                       : static_cast<uint32_t>(page_table_ptr_u16[virtual_block_id]);
                const uint32_t block_start_id = physical_block_id * num_heads * block_size_t * Wt;
                const uint32_t block_row_tile = (update_idx % block_size) / TILE_HEIGHT;
                const uint32_t block_offset = block_row_tile * Wt;
                cache_id = block_start_id + block_offset;

                // Page-table pages consumed; pop the same count waited above to balance the buffer.
                dfb_page_table.pop_front(num_pages_to_read);
            }
#else
            {
                const uint32_t cache_batch_tile_offset = my_batch_idx * cache_batch_num_tiles;
                const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
                cache_id = cache_start_id;
            }
#endif
            cache_tile_offset_B = update_idx % TILE_HEIGHT * Wbytes;
        }
        // The index value is consumed on both the skip and update paths; the reader pushes
        // the index buffer unconditionally, so pop it here (outside the skip branch) to balance
        // the wait.
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
        // Wait on compute to untilize a block. Update that block in SRAM.
        dfb_untilized_cache.wait_front(Wt);
        dfb_untilized_cache2.reserve_back(Wt);

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
        // send signal to receiver core that we are done using the input buffer
        Semaphore(sem::in0_seq).up(noc, send_core_x, send_core_y, 1);
        // Drain the non-posted atomic before kernel_main returns. .up() lowers to a non-posted
        // noc_semaphore_inc tracked by a separate atomic counter that the noc.async_write_barrier()
        // in the head loop above does NOT drain, so without this the kernel exits with the readiness
        // atomic still in flight, which is an inter-kernel NOC race (Watcher NOC-idle assert).
        // Mirrors the sibling writer_update_cache_interleaved_start_id.cpp.
        noc.async_atomic_barrier();
    }
}
