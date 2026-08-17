// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

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

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_page_table = get_compile_time_arg_val(1);
    constexpr uint32_t num_heads = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_of_work_per_head = get_compile_time_arg_val(3);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t log2_page_table_stick_size = get_compile_time_arg_val(6);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(7);

    // Compile-time args for optional batch_idx_tensor.
    // When use_batch_idx_tensor is true, runtime arg 4 is the address of a
    // 1D int tensor with `batch_idx_num_elements` entries:
    //   1            -> single-batch (legacy) path; input_tensor.shape[0] == 1.
    //   input_batch  -> batched path; one batch_idx per input batch row,
    //                   selected per row via `num_blocks_per_batch`.
    constexpr bool use_batch_idx_tensor = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t cb_id_batch_idx = get_compile_time_arg_val(9);
    constexpr uint32_t batch_idx_stick_size = get_compile_time_arg_val(10);  // per-element size, e.g. 4 for uint32
    constexpr uint32_t batch_idx_num_elements = get_compile_time_arg_val(11);
    constexpr uint32_t num_blocks_per_batch = get_compile_time_arg_val(12);  // num_heads * input_seq_len_t
    constexpr uint32_t capacity_t = get_compile_time_arg_val(13);
    // Optional valid_seq_len tensor: when use_valid_seq_len is true, runtime arg 6 is
    // the address of a 1-element int tensor holding the block-aligned real fill length
    // (in tokens). It restricts the bounded ring window to end at valid_seq_len rather
    // than the padded input end (see below).
    constexpr bool use_valid_seq_len = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t cb_id_valid_seq_len = get_compile_time_arg_val(15);
    constexpr uint32_t valid_seq_len_stick_size = get_compile_time_arg_val(16);
    constexpr bool batched_fill = batch_idx_num_elements > 1;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t page_table_addr = get_arg_val<uint32_t>(1);
    uint32_t start_row_num = get_arg_val<uint32_t>(2);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    // Arg 4 is either batch_idx_tensor_addr or batch_idx_fallback scalar
    uint32_t batch_arg = get_arg_val<uint32_t>(4);
    uint32_t noop = get_arg_val<uint32_t>(5);
    uint32_t valid_seq_len_addr = get_arg_val<uint32_t>(6);

    if (noop == 1) {
        return;  // Early exit, no work done
    }

    constexpr auto s0_args = TensorAccessorArgs<17>();
    constexpr auto page_table_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    constexpr auto batch_idx_tensor_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto valid_seq_len_tensor_args =
        TensorAccessorArgs<batch_idx_tensor_args.next_compile_time_args_offset()>();

    CircularBuffer cb_in(cb_id_in);
    CircularBuffer cb_page_table(cb_id_page_table);
    CircularBuffer cb_batch_idx(cb_id_batch_idx);
    CircularBuffer cb_valid_seq_len(cb_id_valid_seq_len);

    // Resolve batch_idx source. For use_batch_idx_tensor=true we load the
    // (small) 1D tensor into an L1 CB once so per-row lookups stay local.
    // For use_batch_idx_tensor=false the scalar fallback in arg 4 is used.
    volatile tt_l1_ptr uint32_t* batch_idx_arr = nullptr;
    uint32_t scalar_batch_idx = 0;
    if constexpr (use_batch_idx_tensor) {
        const auto batch_idx_gen = TensorAccessor(batch_idx_tensor_args, batch_arg);
        cb_batch_idx.reserve_back(1);
        const uint32_t batch_idx_cb_wr_ptr = cb_batch_idx.get_write_ptr();
        // The tensor is a contiguous 1D int (uint32/int32) tensor in DRAM with
        // `batch_idx_num_elements` entries; one TensorAccessor stick covers it.
        noc.async_read(
            batch_idx_gen,
            CoreLocalMem<uint32_t>(batch_idx_cb_wr_ptr),
            batch_idx_stick_size * batch_idx_num_elements,
            {.page_id = 0},
            {});
        noc.async_read_barrier();
        batch_idx_arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_idx_cb_wr_ptr);
    } else {
        scalar_batch_idx = batch_arg;
    }

    // Resolve the optional valid_seq_len. When present, `effective_end` (in tiles)
    // caps the surviving window at the last real token instead of the padded end.
    uint32_t effective_end = num_blocks_of_work_per_head;
    if constexpr (use_valid_seq_len) {
        const auto valid_gen = TensorAccessor(valid_seq_len_tensor_args, valid_seq_len_addr);
        cb_valid_seq_len.reserve_back(1);
        const uint32_t valid_cb_wr_ptr = cb_valid_seq_len.get_write_ptr();
        noc.async_read(
            valid_gen, CoreLocalMem<uint32_t>(valid_cb_wr_ptr), valid_seq_len_stick_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        const uint32_t valid_seq_len_tokens = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_cb_wr_ptr);
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

    const uint32_t tile_bytes = get_tile_size(cb_id_in);
    const DataFormat data_format = get_dataformat(cb_id_in);

    const auto out_gen = TensorAccessor(s0_args, dst_addr);
    const auto page_table_gen = TensorAccessor(page_table_args, page_table_addr);

    cb_page_table.reserve_back(1);
    const uint32_t page_table_cb_wr_ptr = cb_page_table.get_write_ptr();
    volatile tt_l1_ptr uint32_t* page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);

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
        // window. The input CB still has to be drained so the reader doesn't stall,
        // but no NOC writes go out.
        if (seq_tile_id < skip_tiles || seq_tile_id >= effective_end) {
            cb_in.wait_front(Wt);
            cb_in.pop_front(Wt);
            continue;
        }

        uint32_t batch_idx;
        if constexpr (use_batch_idx_tensor) {
            batch_idx = batch_idx_arr[batched_fill ? cur_batch : 0];
        } else {
            batch_idx = scalar_batch_idx;
        }

        // Reload page_table row only on batch boundary.
        if (batch_idx != cached_batch) {
            noc.async_read(
                page_table_gen,
                CoreLocalMem<uint32_t>(page_table_cb_wr_ptr),
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
            // Block should be skipped. Consume the input tiles from the CB and discard.
            cb_in.wait_front(Wt);
            cb_in.pop_front(Wt);
        } else {
            // Valid block, proceed with writing.
            cb_in.wait_front(Wt);
            uint32_t l1_read_addr = cb_in.get_read_ptr();
            for (uint32_t w = 0; w < Wt; ++w) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(l1_read_addr), out_gen, tile_bytes, {}, {.page_id = physical_tile_id});
                l1_read_addr += tile_bytes;
                physical_tile_id += 1;
            }
            noc.async_write_barrier();
            cb_in.pop_front(Wt);
        }
    }
}
