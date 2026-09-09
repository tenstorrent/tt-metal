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

// Whether each optional tensor lives in DRAM. When it does not, its buffer is built on the tensor's
// own resident shard and the kernel reads it in place; only the DRAM case NoC-reads through an
// accessor, which is why the accessor construction below is the part behind the preprocessor gate.
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

    const auto cache_start_id = get_arg(args::cache_start_id);
    const auto my_batch_idx = get_arg(args::my_batch_idx);
    const bool wait_to_start_signal = get_arg(args::wait_to_start) == 1;

    constexpr auto cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto log_base_2_of_page_size = get_arg(args::log_base_2_of_page_size);
    constexpr auto index_stick_size_B = get_arg(args::index_stick_size_B);

    // paged_cache args
    constexpr auto num_heads = get_arg(args::num_heads);
    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto block_size_t = get_arg(args::block_size_t);
    constexpr auto max_blocks_per_seq = get_arg(args::max_blocks_per_seq);
    constexpr auto log2_page_table_stick_size = get_arg(args::log2_page_table_stick_size);
    constexpr auto page_table_stick_size = get_arg(args::page_table_stick_size);

    constexpr auto St = get_arg(args::St);
    constexpr auto batch_size = get_arg(args::batch_size);

    constexpr uint32_t head_offset_t = Wt * St;

    // Which of the two inputs this instance serves is structural: the factory places one reader per
    // input shard grid, and each binds only its own input buffer.
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_cache(dfb::cache);
#ifdef USE_INDEX_TENSOR
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    // Kick off compute
    dfb_input.reserve_back(1);
    dfb_input.push_back(1);

    const uint32_t cache_tile_bytes = dfb_cache.get_tile_size();

    constexpr uint32_t TILE_HEIGHT = 32;

    uint32_t cache_id = cache_start_id;

    const auto s0 = TensorAccessor(tensor::cache);

    bool skip_update = false;

#ifdef USE_INDEX_TENSOR
    {
        dfb_index.reserve_back(1);
        uint32_t index_wr_ptr = dfb_index.get_write_ptr();
#ifdef INDEX_IS_DRAM
        {
            const auto addrg = TensorAccessor(tensor::index);
            // index_tensor has one page to read
            noc.async_read(addrg, CoreLocalMem<uint32_t>(index_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
            noc.async_read_barrier();
        }
#endif
        dfb_index.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_wr_ptr);

        const uint32_t update_idx = index_ptr[my_batch_idx];
        if (update_idx == (uint32_t)-1) {
            // Passing update_idx = -1 tells us to skip update for this user
            skip_update = true;
        } else {
#ifdef IS_PAGED_CACHE
            {
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
                volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
                volatile tt_l1_ptr uint16_t* page_table_ptr_u16 = nullptr;

                if constexpr (page_table_is_dram) {
                    page_table_ptr_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_wr_ptr);
                } else {
                    page_table_ptr_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_wr_ptr);
                }

                const uint32_t virtual_block_id = update_idx / block_size;
                const uint32_t physical_block_id = (page_table_is_dram)
                                                       ? page_table_ptr_u32[virtual_block_id]
                                                       : static_cast<uint32_t>(page_table_ptr_u16[virtual_block_id]);

                const uint32_t block_start_id = physical_block_id * num_heads * block_size_t * Wt;
                const uint32_t block_row_tile = (update_idx % block_size) / TILE_HEIGHT;
                const uint32_t block_offset = block_row_tile * Wt;
                cache_id = block_start_id + block_offset;
            }
#else
            {
                const uint32_t cache_batch_tile_offset = my_batch_idx * cache_batch_num_tiles;
                const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
                cache_id = cache_start_id;
            }
#endif
        }
    }
#endif

    if (wait_to_start_signal) {
        // wait for signal from writer that it has finished using the input buffer
        Semaphore receiver_sem(sem::in0_seq);
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
