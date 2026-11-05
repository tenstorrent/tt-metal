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
    auto cache_tile_offset_B = get_arg(args::cache_tile_offset_B);
    const auto my_batch_idx = get_arg(args::my_batch_idx);
    const bool send_signal = get_arg(args::send_signal) == 1;
    const auto send_core_x = get_arg(args::send_core_x);
    const auto send_core_y = get_arg(args::send_core_y);
    // Heads [head_start, head_start + num_local_heads) of this user (see the reader).
    const auto head_start = get_arg(args::head_start);
    const auto num_local_heads = get_arg(args::num_local_heads);

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

#ifndef ROW_PATCH
    // `dfb::cache` is this kernel's name for the *output* buffer, which compute fills with the
    // retilized block; the reader's same-named handle is the cache buffer it read from DRAM.
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_untilized_cache(dfb::untilized_cache);
    DataflowBuffer dfb_untilized_cache2(dfb::untilized_cache2);
    DataflowBuffer dfb_untilized_input(dfb::untilized_input);
#endif
#ifdef USE_INDEX_TENSOR
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    // get_entry_size(): this DFB's live per-program entry size. get_tile_size() reads a global descriptor
    // array that intervening ops clobber on Quasar DM kernels (stale -> wrong size -> DRAM over/under-run on
    // a later invocation; cf. fill_cache). Byte-identical to get_tile_size() on WH/BH.
#ifndef ROW_PATCH
    const uint32_t cache_tile_bytes = dfb_cache.get_entry_size();
#endif

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

#ifdef ROW_PATCH
    {
        DataflowBuffer dfb_patch_src(dfb::patch_src);
        DataflowBuffer dfb_scratch(dfb::patch_scratch);
        dfb_patch_src.wait_front(Wt);
        const uint32_t src_base = dfb_patch_src.get_read_ptr();
        const uint32_t src_tile_bytes = dfb_patch_src.get_entry_size();
        const uint8_t noc_id = noc.get_noc_id();

        const uint32_t row = cache_tile_offset_B / Wbytes;  // 0..31
        const uint32_t dface = row_patch::first_face(row), drow = row % 16;
        const uint32_t prefix = row_patch::prefix_bytes(row);
        const uint32_t entry = dfb_scratch.get_entry_size();
        const uint32_t mant_dst0 = 64 + dface * 256 + drow * 16;
        const uint32_t mant_dst1 = mant_dst0 + 256;

        const uint32_t first_id = cache_id + head_start * head_offset_t;
        for (uint32_t h0 = 0; h0 < num_local_heads; h0 += row_patch::kHeadsPerGroup) {
            const uint32_t heads = (num_local_heads - h0) < row_patch::kHeadsPerGroup ? (num_local_heads - h0)
                                                                                     : row_patch::kHeadsPerGroup;
            dfb_scratch.wait_front(heads * Wt);
            if (!skip_update) {
                uint32_t l1 = dfb_scratch.get_read_ptr();
                for (uint32_t h = 0; h < heads; ++h) {
                    const uint32_t head = head_start + h0 + h;  // its row in the packed input tiles
                    const uint32_t sface = row_patch::first_face(head), srow = head % 16;
                    for (uint32_t t = 0; t < Wt; ++t, l1 += entry) {
                        const uint32_t src = src_base + t * src_tile_bytes;
                        volatile tt_l1_ptr uint8_t* src_exp = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
                        volatile tt_l1_ptr uint8_t* dst_exp = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1);
                        dst_exp[dface * 16 + drow] = src_exp[sface * 16 + srow];
                        dst_exp[(dface + 1) * 16 + drow] = src_exp[(sface + 1) * 16 + srow];
                        volatile tt_l1_ptr uint32_t* s0w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            src + 64 + sface * 256 + srow * 16);
                        volatile tt_l1_ptr uint32_t* s1w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            src + 64 + (sface + 1) * 256 + srow * 16);
                        volatile tt_l1_ptr uint32_t* d0w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + mant_dst0);
                        volatile tt_l1_ptr uint32_t* d1w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + mant_dst1);
                        for (uint32_t w = 0; w < 4; ++w) {
                            d0w[w] = s0w[w];
                            d1w[w] = s1w[w];
                        }
                        const uint64_t tile = s0.get_noc_addr(first_id + (h0 + h) * head_offset_t + t, 0, noc_id);
                        noc_async_write(l1, tile, prefix, noc_id);
                    }
                }
                noc_async_writes_flushed(noc_id);
            }
            dfb_scratch.pop_front(heads * Wt);
        }
        noc_async_write_barrier(noc_id);
        dfb_patch_src.pop_front(Wt);
    }
#else
    dfb_untilized_input.wait_front(Wt);  // input tensor
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    uint32_t input_l1_read_addr = dfb_untilized_input.get_read_ptr() + head_start * Wbytes;
    UnicastEndpoint local_src;

    cache_id += head_start * head_offset_t;
    for (uint32_t cur_head = 0; cur_head < num_local_heads; ++cur_head) {
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
#endif

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
