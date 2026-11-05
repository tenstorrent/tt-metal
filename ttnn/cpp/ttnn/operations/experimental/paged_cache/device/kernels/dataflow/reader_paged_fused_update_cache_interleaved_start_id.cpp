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

// bfp8_b row patch (ROW_PATCH). A bfp8_b tile is 64 exponent bytes (byte face * 16 + row: one shared exponent per
// 16-value face row) followed by four 16 x 16 faces of 1-byte mantissas (face * 256 + row * 16). Tile row r lives in
// faces 0 and 1 (r < 16) or 2 and 3, at face row r % 16, so updating one row touches two exponent bytes and two
// 16-byte mantissa rows per tile. The reader fetches each tile from its start to the end of the 64-byte granule that
// holds the row in the second face (one transaction: 192-576 bytes for rows 0-15, 704-1088 for rows 16-31); the
// writer patches the row from the packed input and writes the same bytes back (one transaction).
namespace row_patch {
constexpr uint32_t kHeadsPerGroup = 4;
FORCE_INLINE uint32_t first_face(uint32_t row) { return row < 16 ? 0 : 2; }
// Bytes from the start of the tile through the granule holding cache row `row` in its second face.
FORCE_INLINE uint32_t prefix_bytes(uint32_t row) { return 64 + (first_face(row) + 1) * 256 + ((row % 16) / 4) * 64 + 64; }
}  // namespace row_patch

void kernel_main() {
    Noc noc;

    const auto has_work = get_arg(args::has_work);
    if (!has_work) {
        return;
    }

    const auto cache_start_id = get_arg(args::cache_start_id);
#ifdef ROW_PATCH
    uint32_t cache_row = get_arg(args::cache_row);  // update_idx % 32; the index tensor path overrides it
#endif
    const auto my_batch_idx = get_arg(args::my_batch_idx);
    const bool wait_to_start_signal = get_arg(args::wait_to_start) == 1;
    // This core updates heads [head_start, head_start + num_local_heads) of its user. A user's heads may be
    // split over several cores; only the core holding the user's input shard reads it in place.
    const auto head_start = get_arg(args::head_start);
    const auto num_local_heads = get_arg(args::num_local_heads);

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
#ifndef ROW_PATCH
    DataflowBuffer dfb_cache(dfb::cache);
#endif
#ifdef USE_INDEX_TENSOR
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    // Kick off compute
    dfb_input.reserve_back(Wt);
#ifdef REMOTE_INPUT
    {
        // Helper core: copy this user's input tiles from the core that holds its shard.
        const auto input_accessor = TensorAccessor(tensor::input);
        uint32_t input_l1_write_addr = dfb_input.get_write_ptr();
        const uint32_t input_tile_bytes = dfb_input.get_entry_size();
        for (uint32_t t = 0; t < Wt; ++t) {
            noc.async_read(
                input_accessor,
                CoreLocalMem<uint32_t>(input_l1_write_addr),
                input_tile_bytes,
                {.page_id = my_batch_idx * Wt + t},
                {});
            input_l1_write_addr += input_tile_bytes;
        }
        noc.async_read_barrier();
    }
#endif
    dfb_input.push_back(Wt);

    // get_entry_size(): this DFB's live per-program entry size. get_tile_size() reads a global descriptor
    // array that intervening ops clobber on Quasar DM kernels (stale -> wrong size -> DRAM over/under-run on
    // a later invocation; cf. fill_cache). Byte-identical to get_tile_size() on WH/BH.
#ifndef ROW_PATCH
    const uint32_t cache_tile_bytes = dfb_cache.get_entry_size();
    const auto s0 = TensorAccessor(tensor::cache);
#endif

    constexpr uint32_t TILE_HEIGHT = 32;

    uint32_t cache_id = cache_start_id;

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
#ifdef ROW_PATCH
        cache_row = update_idx % TILE_HEIGHT;
#endif
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

#ifdef ROW_PATCH
    {
        DataflowBuffer dfb_scratch(dfb::patch_scratch);
        const auto cache = TensorAccessor(tensor::cache);
        const uint8_t noc_id = noc.get_noc_id();
        const uint32_t prefix = row_patch::prefix_bytes(cache_row);
        const uint32_t entry = dfb_scratch.get_entry_size();
        cache_id += head_start * head_offset_t;
        for (uint32_t h0 = 0; h0 < num_local_heads; h0 += row_patch::kHeadsPerGroup) {
            const uint32_t heads = (num_local_heads - h0) < row_patch::kHeadsPerGroup ? (num_local_heads - h0)
                                                                                     : row_patch::kHeadsPerGroup;
            dfb_scratch.reserve_back(heads * Wt);
            if (!skip_update) {
                uint32_t l1 = dfb_scratch.get_write_ptr();
                for (uint32_t h = 0; h < heads; ++h) {
                    for (uint32_t t = 0; t < Wt; ++t, l1 += entry) {
                        noc_async_read(
                            cache.get_noc_addr(cache_id + (h0 + h) * head_offset_t + t, 0, noc_id), l1, prefix, noc_id);
                    }
                }
                noc_async_read_barrier(noc_id);
            }
            dfb_scratch.push_back(heads * Wt);
        }
    }
#else
    cache_id += head_start * head_offset_t;
    for (uint32_t cur_head = 0; cur_head < num_local_heads; ++cur_head) {
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
#endif
}
