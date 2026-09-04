// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

    const bool has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }
    const bool is_input1 = get_arg(args::is_input1);

    const uint32_t cache_start_id = get_arg(args::cache_start_id);
    const uint32_t my_batch_idx = get_arg(args::my_batch_idx);
    const bool wait_to_start_signal = get_arg(args::wait_to_start) == 1;

    constexpr bool index_is_dram = get_arg(args::index_is_dram) == 1;
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
    constexpr uint32_t page_table_is_dram = get_arg(args::page_table_is_dram);

    const uint32_t St = get_arg(args::St);
    constexpr uint32_t batch_size = get_arg(args::batch_size);

    constexpr uint32_t head_offset_t = Wt * St;

    // This core services exactly one of the two fused inputs, chosen at runtime. Both input DFBs are
    // bound to this kernel -- a kernel cannot touch a DFB it has not bound -- and the runtime
    // `is_input1` arg picks which one to drive. A ternary over the two tokens is enough here:
    // DFBBindingToken carries its identity in a runtime member, so both tokens share one type.
    //
    // Quasar-uplift debt: DFB placement is derived from the bindings, so each input DFB is now
    // configured over the whole kernel grid rather than only its own input's shard cores, and on the
    // other half it carries the sibling input's borrowed base address. Inert on WH/BH -- the
    // `is_input1` guard means those nodes never touch it, and a borrowed DFB allocates nothing, so
    // there is nothing for it to collide with. On Gen2 a DFB's hardware footprint varies with its
    // endpoint configuration and a borrowed DFB over a tensor holding no shard on the node is
    // meaningless, so this wants revisiting there.
    const DFBBindingToken input_token = is_input1 ? dfb::src1 : dfb::src2;
    DataflowBuffer dfb_input(input_token);
    // dfb::cache holds the cache tiles this reader pulls in from the cache tensor, for compute.
    DataflowBuffer dfb_cache(dfb::cache);
#ifdef USE_INDEX_TENSOR
    // Bound only in index-tensor mode: the reader fills it from the index tensor and the writer reads
    // the same value back out of it. On the L1-sharded path it is a borrowed view over the index
    // tensor itself, so there is nothing to transfer into it.
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    // Bound only for a paged cache; likewise reader-filled and writer-read, and likewise borrowed
    // when the page table is L1-sharded.
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    // Kick off compute
    dfb_input.reserve_back(1);
    dfb_input.push_back(1);

    const uint32_t cache_tile_bytes = dfb_cache.get_tile_size();

    constexpr uint32_t TILE_HEIGHT = 32;

    uint32_t cache_id = cache_start_id;

    bool skip_update = false;

#ifdef USE_INDEX_TENSOR
    {
        const auto addrg = TensorAccessor(tensor::index);

        dfb_index.reserve_back(1);
        uint32_t index_dfb_wr_ptr = dfb_index.get_write_ptr();
        // index_tensor has one page to read
        if constexpr (index_is_dram) {
            noc.async_read(addrg, CoreLocalMem<uint32_t>(index_dfb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
            noc.async_read_barrier();
        }

        dfb_index.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_dfb_wr_ptr);

        const uint32_t update_idx = index_ptr[my_batch_idx];

        if (update_idx == (uint32_t)-1) {
            // Passing update_idx = -1 tells us to skip update for this user
            skip_update = true;
        } else {
#ifdef IS_PAGED_CACHE
            uint32_t num_pages_to_read = page_table_is_dram ? 1 : batch_size;

            dfb_page_table.reserve_back(num_pages_to_read);
            uint32_t page_table_dfb_wr_ptr = dfb_page_table.get_write_ptr();

            if constexpr (page_table_is_dram) {
                const auto page_table_gen = TensorAccessor(tensor::page_table);
                noc.async_read(
                    page_table_gen,
                    CoreLocalMem<uint32_t>(page_table_dfb_wr_ptr),
                    page_table_stick_size,
                    {.page_id = my_batch_idx},
                    {});
                noc.async_read_barrier();
            } else {
                page_table_dfb_wr_ptr += my_batch_idx * page_table_stick_size;
            }
            dfb_page_table.push_back(num_pages_to_read);
            // DRAM uses uint32 entries; L1-sharded page table uses uint16 entries
            volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
            volatile tt_l1_ptr uint16_t* page_table_ptr_u16 = nullptr;
            if constexpr (page_table_is_dram) {
                page_table_ptr_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_dfb_wr_ptr);
            } else {
                page_table_ptr_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_dfb_wr_ptr);
            }

            const uint32_t virtual_block_id = update_idx / block_size;
            const uint32_t physical_block_id = (page_table_is_dram)
                                                   ? page_table_ptr_u32[virtual_block_id]
                                                   : static_cast<uint32_t>(page_table_ptr_u16[virtual_block_id]);

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

    // The cache tensor this core reads from is picked by the same runtime `is_input1` arg, but the
    // tensor channel cannot use the ternary the DFB channel above does: host codegen emits a distinct
    // TensorBindingToken *type* per binding (its CTA and CRTA slot offsets are template parameters,
    // and two bindings on one kernel necessarily occupy different slots). So the per-head loop is a
    // generic lambda, instantiated once per accessor type and selected once, here. Inside the loop
    // the accessor is a concrete type and fully inlined.
    const auto read_cache_blocks = [&](const auto& s0) {
        for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
            dfb_cache.reserve_back(Wt);
            if (!skip_update) {
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
            }
            dfb_cache.push_back(Wt);

            cache_id += head_offset_t;
        }
    };

    if (is_input1) {
        read_cache_blocks(TensorAccessor(tensor::cache1));
    } else {
        read_cache_blocks(TensorAccessor(tensor::cache2));
    }
}
