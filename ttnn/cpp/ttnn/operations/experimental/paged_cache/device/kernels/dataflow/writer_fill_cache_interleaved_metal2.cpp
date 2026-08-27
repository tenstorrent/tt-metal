// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Define the sentinel value for a page table entry that indicates a skip.
constexpr uint32_t SKIP_PAGE_TABLE_ENTRY = (uint32_t)-1;

// Tile height in rows (Blackhole/Wormhole tiles are 32x32).
constexpr uint32_t TILE_H = 32;

template <uint32_t num_heads, uint32_t block_size_t, uint32_t Wt>
uint32_t virtual_seq_tile_id_to_physical_tile_id(
    uint32_t seq_tile_idx, uint32_t cur_head, volatile tt_l1_ptr const uint32_t* const page_table_ptr) {
    // Given some index in the sequence tiles in range [0, max_seq_len_t]
    // Return the physical tile id for that tile row, or SKIP_PAGE_TABLE_ENTRY if block is skipped
    constexpr uint32_t block_stride = num_heads * block_size_t * Wt;
    const uint32_t head_offset = cur_head * block_size_t * Wt;

    const uint32_t virtual_block = seq_tile_idx / block_size_t;
    const uint32_t physical_block = page_table_ptr[virtual_block];

    if (physical_block == SKIP_PAGE_TABLE_ENTRY) {
        return SKIP_PAGE_TABLE_ENTRY;  // Return sentinel to indicate skip
    }

    const uint32_t block_row_offset = seq_tile_idx % block_size_t;
    const uint32_t block_offset = block_row_offset * Wt;
    return physical_block * block_stride + head_offset + block_offset;
}

void kernel_main() {
    Noc noc;

    constexpr uint32_t num_heads = get_arg(args::num_heads);
    constexpr uint32_t num_blocks_of_work_per_head = get_arg(args::num_blocks_of_work_per_head);
    constexpr uint32_t block_size_t = get_arg(args::block_size_t);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t log2_page_table_stick_size = get_arg(args::log2_page_table_stick_size);
    constexpr uint32_t page_table_stick_size = get_arg(args::page_table_stick_size);

    // Optional batch_idx_tensor.
    // When USE_BATCH_IDX_TENSOR is defined, tensor::batch_idx binds a 1D int tensor with
    // `batch_idx_num_elements` entries:
    //   1            -> single-batch (legacy) path; input_tensor.shape[0] == 1.
    //   input_batch  -> batched path; one batch_idx per input batch row,
    //                   selected per row via `num_blocks_per_batch`.
    constexpr uint32_t batch_idx_stick_size = get_arg(args::batch_idx_stick_size);  // per-element size, e.g. 4 for
                                                                                   // uint32
    constexpr uint32_t batch_idx_num_elements = get_arg(args::batch_idx_num_elements);
    constexpr uint32_t num_blocks_per_batch = get_arg(args::num_blocks_per_batch);  // num_heads * input_seq_len_t
    constexpr uint32_t capacity_t = get_arg(args::capacity_t);
    // Optional valid_seq_len tensor: when USE_VALID_SEQ_LEN is defined, tensor::valid_seq_len binds
    // a 1-element int tensor holding the block-aligned real fill length (in tokens). It restricts
    // the bounded ring window to end at valid_seq_len rather than the padded input end (see below).
    constexpr uint32_t valid_seq_len_stick_size = get_arg(args::valid_seq_len_stick_size);
    constexpr bool batched_fill = batch_idx_num_elements > 1;

    uint32_t start_row_num = get_arg(args::start_row_num);
    uint32_t num_rows = get_arg(args::num_rows);
    uint32_t noop = get_arg(args::noop);

    if (noop == 1) {
        return;  // Early exit, no work done
    }

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_page_table(dfb::page_table);
#ifdef USE_BATCH_IDX_TENSOR
    DataflowBuffer dfb_batch_idx(dfb::batch_idx);
#endif
#ifdef USE_VALID_SEQ_LEN
    DataflowBuffer dfb_valid_seq_len(dfb::valid_seq_len);
#endif

    // Resolve batch_idx source. With USE_BATCH_IDX_TENSOR we load the
    // (small) 1D tensor into an L1 DFB once so per-row lookups stay local.
    // Otherwise the scalar fallback runtime arg is used.
    volatile tt_l1_ptr uint32_t* batch_idx_arr = nullptr;
    uint32_t scalar_batch_idx = 0;
#ifdef USE_BATCH_IDX_TENSOR
    {
        const auto batch_idx_gen = TensorAccessor(tensor::batch_idx);
        dfb_batch_idx.reserve_back(1);
        const uint32_t batch_idx_dfb_wr_ptr = dfb_batch_idx.get_write_ptr();
        // The tensor is a contiguous 1D int (uint32/int32) tensor in DRAM with
        // `batch_idx_num_elements` entries; one TensorAccessor stick covers it.
        noc.async_read(
            batch_idx_gen,
            CoreLocalMem<uint32_t>(batch_idx_dfb_wr_ptr),
            batch_idx_stick_size * batch_idx_num_elements,
            {.page_id = 0},
            {});
        noc.async_read_barrier();
        batch_idx_arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_idx_dfb_wr_ptr);
    }
#else
    scalar_batch_idx = get_arg(args::batch_idx_fallback);
#endif

    // Resolve the optional valid_seq_len. When present, `effective_end` (in tiles)
    // caps the surviving window at the last real token instead of the padded end.
    uint32_t effective_end = num_blocks_of_work_per_head;
#ifdef USE_VALID_SEQ_LEN
    {
        const auto valid_gen = TensorAccessor(tensor::valid_seq_len);
        dfb_valid_seq_len.reserve_back(1);
        const uint32_t valid_dfb_wr_ptr = dfb_valid_seq_len.get_write_ptr();
        noc.async_read(
            valid_gen, CoreLocalMem<uint32_t>(valid_dfb_wr_ptr), valid_seq_len_stick_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        const uint32_t valid_seq_len_tokens = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_dfb_wr_ptr);
        // Round the real length up to a whole tile (TILE_HEIGHT == 32, >> 5) and then
        // up to a whole block_size_t: the surviving ring window [effective_end -
        // capacity_t, effective_end) must be block-aligned so the wrapped page_table
        // lookup preserves intra-block layout. This equals the host cap's
        // ceil(valid/block)*block, so the kernel can take the raw valid length.
        uint32_t valid_tiles = (valid_seq_len_tokens + (TILE_H - 1)) >> 5;
        if (block_size_t > 0) {
            valid_tiles = ((valid_tiles + block_size_t - 1) / block_size_t) * block_size_t;
        }
        if (valid_tiles > 0 && valid_tiles < num_blocks_of_work_per_head) {
            effective_end = valid_tiles;
        }
    }
#endif

    const uint32_t tile_bytes = dfb_in.get_tile_size();

    const auto out_gen = TensorAccessor(tensor::out);
    const auto page_table_gen = TensorAccessor(tensor::page_table);

    dfb_page_table.reserve_back(1);
    const uint32_t page_table_dfb_wr_ptr = dfb_page_table.get_write_ptr();
    volatile tt_l1_ptr uint32_t* page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_dfb_wr_ptr);

    // Cache the last batch for which page_table was loaded. Legacy path loads
    // once on the first row (cache miss) and then hits for all remaining rows;
    // batched path re-loads on batch boundaries within this core's row range.
    uint32_t cached_batch = (uint32_t)-1;
    // Bounded sliding-window cache (capacity_t > 0): only the last capacity_t tiles
    // of each head's input range will survive in the cache (earlier tiles would map
    // to wrapped slots that get overwritten by later writes). Compute the skip
    // count once so we can consume those earlier input tiles without committing
    // them to DRAM — a strict bandwidth win for prefills longer than the bounded
    // capacity. For prefills <= capacity_t this is 0 and the legacy path runs.
    // With an optional valid_seq_len cap, the ring window ends at effective_end
    // (== num_blocks_of_work_per_head when uncapped). skip_tiles drops the earliest
    // tiles that a later wrapped write would overwrite anyway, so that exactly the
    // last capacity_t tiles *ending at effective_end* survive — the last real tokens,
    // not the padding tail. Tiles at/after effective_end are padding and are skipped
    // in the loop below.
    const uint32_t skip_tiles = (capacity_t > 0 && effective_end > capacity_t) ? (effective_end - capacity_t) : 0;

    for (uint32_t row_id = start_row_num; row_id < start_row_num + num_rows; ++row_id) {
        // Decode row_id → (cur_batch, cur_head, seq_tile_id).
        // Input layout: [input_batch, num_heads, input_seq_len_t]
        //   row_id = cur_batch * num_blocks_per_batch
        //            + cur_head * num_blocks_of_work_per_head
        //            + seq_tile_id
        // On the legacy path (batched_fill == false) cur_batch is always 0
        // and the per-batch arithmetic is elided.
        uint32_t cur_batch;
        uint32_t row_within_batch;
        if constexpr (batched_fill) {
            cur_batch = row_id / num_blocks_per_batch;
            row_within_batch = row_id - cur_batch * num_blocks_per_batch;
        } else {
            cur_batch = 0;
            row_within_batch = row_id;
        }
        const uint32_t cur_head = row_within_batch / num_blocks_of_work_per_head;
        uint32_t seq_tile_id = row_within_batch % num_blocks_of_work_per_head;

        // Drop the early-prefill tiles whose final slot would be overwritten by a
        // later iteration anyway, and (with a valid_seq_len cap) the trailing padding
        // tiles at/after effective_end that would otherwise wrap over the real recent
        // window. The input DFB still has to be drained so the reader doesn't stall,
        // but no NOC writes go out.
        if (seq_tile_id < skip_tiles || seq_tile_id >= effective_end) {
            dfb_in.wait_front(Wt);
            dfb_in.pop_front(Wt);
            continue;
        }

        uint32_t batch_idx;
#ifdef USE_BATCH_IDX_TENSOR
        batch_idx = batch_idx_arr[batched_fill ? cur_batch : 0];
#else
        batch_idx = scalar_batch_idx;
#endif

        // Reload page_table row only on batch boundary.
        if (batch_idx != cached_batch) {
            noc.async_read(
                page_table_gen,
                CoreLocalMem<uint32_t>(page_table_dfb_wr_ptr),
                page_table_stick_size,
                {.page_id = batch_idx},
                {});
            noc.async_read_barrier();
            cached_batch = batch_idx;
        }

        // Bounded sliding-window cache: wrap the virtual tile index into the bounded
        // capacity before the page_table lookup. capacity_t is a multiple of
        // block_size_t (validated), so the wrap preserves intra-block layout.
        if constexpr (capacity_t > 0) {
            seq_tile_id %= capacity_t;
        }
        uint32_t physical_tile_id =
            virtual_seq_tile_id_to_physical_tile_id<num_heads, block_size_t, Wt>(seq_tile_id, cur_head, page_table_ptr);

        if (physical_tile_id == SKIP_PAGE_TABLE_ENTRY) {
            // Block should be skipped. Consume the input tiles from the DFB and discard.
            dfb_in.wait_front(Wt);
            dfb_in.pop_front(Wt);
        } else {
            // Valid block, proceed with writing.
            dfb_in.wait_front(Wt);
            uint32_t l1_read_addr = dfb_in.get_read_ptr();
            for (uint32_t w = 0; w < Wt; ++w) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr), out_gen, tile_bytes, {}, {.page_id = physical_tile_id});
                l1_read_addr += tile_bytes;
                physical_tile_id += 1;
            }
            noc.async_write_barrier();
            dfb_in.pop_front(Wt);
        }
    }
}
