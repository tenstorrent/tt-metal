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

void kernel_main() {
    Noc noc;

    const bool has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }

    const uint32_t cache_start_id = get_arg(args::cache_start_id);
    uint32_t cache_tile_offset_B = get_arg(args::cache_tile_offset_B);
    const uint32_t my_batch_idx = get_arg(args::my_batch_idx);
    const bool send_signal = get_arg(args::send_signal) == 1;
    const uint32_t send_core_x = get_arg(args::send_core_x);
    const uint32_t send_core_y = get_arg(args::send_core_y);
    // Selects which of the two fused cache tensors this core writes. The legacy kernel carried no
    // such arg: it received the chosen cache tensor's base address directly in a runtime-arg slot,
    // and the host wrote input1's address on one core list and input2's on the other. That address
    // slot is gone -- both cache tensors are bound as typed tensor parameters instead -- so the
    // choice itself now has to reach the kernel, and this is it. (The row-major sibling writer
    // already took an is_input1 arg, for the same "which input am I" purpose.)
    const bool is_input1 = get_arg(args::is_input1);

    constexpr uint32_t cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Wbytes = get_arg(args::Wbytes);

    // paged_cache args
    constexpr uint32_t num_heads = get_arg(args::num_heads);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t block_size_t = get_arg(args::block_size_t);
    constexpr uint32_t max_blocks_per_seq = get_arg(args::max_blocks_per_seq);

    constexpr uint32_t St = get_arg(args::St);
    constexpr uint32_t head_offset_t = Wt * St;
    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t page_table_stick_size = get_arg(args::page_table_stick_size);
    constexpr uint32_t page_table_is_dram = get_arg(args::page_table_is_dram);

    constexpr uint32_t TILE_HEIGHT = 32;

    // NOTE: this kernel's `cache` DFB is the *output* buffer (the host binds its output
    // DataflowBufferSpec here) -- it holds the re-tilized cache block this kernel writes back to the
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
        const uint32_t update_idx = index_ptr[my_batch_idx];

        if (update_idx == (uint32_t)-1) {
            // Passing update_idx = -1 tells us to skip update for this user
            skip_update = true;
        } else {
#ifdef IS_PAGED_CACHE
            uint32_t num_pages_to_read = page_table_is_dram ? 1 : batch_size;
            dfb_page_table.wait_front(num_pages_to_read);
            uint32_t page_table_dfb_rd_ptr = dfb_page_table.get_read_ptr();
            if constexpr (!page_table_is_dram) {
                page_table_dfb_rd_ptr += my_batch_idx * page_table_stick_size;
            }
            // DRAM uses uint32 entries; L1-sharded page table uses uint16 entries
            volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
            volatile tt_l1_ptr uint16_t* page_table_ptr_u16 = nullptr;
            if constexpr (page_table_is_dram) {
                page_table_ptr_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_dfb_rd_ptr);
            } else {
                page_table_ptr_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_dfb_rd_ptr);
            }

            const uint32_t virtual_block_id = update_idx / block_size;
            const uint32_t physical_block_id = (page_table_is_dram)
                                                   ? page_table_ptr_u32[virtual_block_id]
                                                   : static_cast<uint32_t>(page_table_ptr_u16[virtual_block_id]);
            const uint32_t block_start_id = physical_block_id * num_heads * block_size_t * Wt;
            const uint32_t block_row_tile = (update_idx % block_size) / TILE_HEIGHT;
            const uint32_t block_offset = block_row_tile * Wt;
            cache_id = block_start_id + block_offset;

            // Page-table pages consumed; pop the same count waited above to balance the DFB.
            dfb_page_table.pop_front(num_pages_to_read);
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

    // The cache tensor this core writes is picked by the runtime `is_input1` arg, but the tensor
    // channel cannot select with a ternary the way the DFB channel can: host codegen emits a distinct
    // TensorBindingToken *type* per binding (its CTA and CRTA slot offsets are template parameters,
    // and two bindings on one kernel necessarily occupy different slots). So the per-head loop is a
    // generic lambda, instantiated once per accessor type and selected once, below. Inside the loop
    // the accessor is a concrete type and fully inlined.
    const auto write_cache_blocks = [&](const auto& s0) {
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
    };

    if (is_input1) {
        write_cache_blocks(TensorAccessor(tensor::cache1));
    } else {
        write_cache_blocks(TensorAccessor(tensor::cache2));
    }

    dfb_untilized_input.pop_front(Wt);

    if (send_signal) {
        // send signal to receiver core that we are done using the input DFB
        Semaphore<>(sem::receiver).up(noc, send_core_x, send_core_y, 1);
        // Drain the non-posted atomic before kernel_main returns. .up() lowers to a non-posted
        // noc_semaphore_inc tracked by a separate atomic counter that noc.async_write_barrier() does
        // NOT drain, so without this the kernel exits with the readiness atomic still in flight -- an
        // inter-kernel NOC race (Watcher NOC-idle assert). Mirrors the sibling
        // writer_update_cache_interleaved_start_id.cpp.
        noc.async_atomic_barrier();
    }
}
