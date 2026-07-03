// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Masked writer for per_token_cast_back. Mirrors the masked reader's block split: the valid compute
// blocks (tile_h scale-blocks each) form a global space = concatenation over this device's experts of
// each expert's ceil(count[e]*blocks_per_row / tile_h) blocks. This core owns the contiguous slice
// [core_id*bpc, core_id*bpc + bpc), maps each block to (expert, row, block_idx), and writes that
// block's produced rows back to their dispatch-buffer pages with one NoC write per bank-contiguous
// run. The reader and writer derive the identical split from the same counts/offsets/core_id/num_cores.
// Garbage rows (outside any region) are never written.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t counts_addr = get_arg_val<uint32_t>(1);
    uint32_t offsets_addr = get_arg_val<uint32_t>(2);
    uint32_t core_id = get_arg_val<uint32_t>(3);
    uint32_t width = get_arg_val<uint32_t>(4);  // H

    constexpr uint32_t cb_out_fp32 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_counts = get_compile_time_arg_val(1);
    constexpr uint32_t cb_offsets = get_compile_time_arg_val(2);
    constexpr uint32_t out_block_bytes = get_compile_time_arg_val(3);  // 128 * out_elem_size
    constexpr uint32_t tile_h = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(5);
    constexpr uint32_t counts_aligned_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t counts_pages = get_compile_time_arg_val(7);
    constexpr uint32_t counter_offset = get_compile_time_arg_val(8);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(9);
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(10);
    constexpr uint32_t num_cores = get_compile_time_arg_val(11);

    constexpr auto dst_args = TensorAccessorArgs<12>();
    constexpr auto counts_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    const auto dst = TensorAccessor(dst_args, dst_addr);
    const auto counts_acc = TensorAccessor(counts_args, counts_addr);
    const auto offsets_acc = TensorAccessor(offsets_args, offsets_addr);

    Noc noc;
    CircularBuffer cb_out_fp32_obj(cb_out_fp32);
    CircularBuffer cb_counts_obj(cb_counts);
    CircularBuffer cb_offsets_obj(cb_offsets);

    const uint32_t blocks_per_row = width >> 7;  // H / 128

    // --- Load counts[] and offsets[] into L1 scratch; index flat by counter_offset. ---
    cb_counts_obj.reserve_back(counts_pages);
    cb_offsets_obj.reserve_back(counts_pages);
    const uint32_t counts_base = cb_counts_obj.get_write_ptr();
    const uint32_t offsets_base = cb_offsets_obj.get_write_ptr();
    for (uint32_t i = 0; i < counts_pages; ++i) {
        noc.async_read(
            counts_acc,
            cb_counts_obj,
            counts_aligned_page_bytes,
            {.page_id = i},
            {.offset_bytes = i * counts_aligned_page_bytes});
        noc.async_read(
            offsets_acc,
            cb_offsets_obj,
            counts_aligned_page_bytes,
            {.page_id = i},
            {.offset_bytes = i * counts_aligned_page_bytes});
    }
    noc.async_read_barrier();
    CoreLocalMem<volatile uint32_t> counts_l1(counts_base);
    CoreLocalMem<volatile uint32_t> offsets_l1(offsets_base);

    // --- Pass 0: clamped per-expert token counts + block prefix sum (same as the reader). ---
    uint32_t cnt_e[experts_per_chip];
    uint32_t prefix[experts_per_chip + 1];
    prefix[0] = 0;
    for (uint32_t e = 0; e < experts_per_chip; ++e) {
        uint32_t start_page = offsets_l1[counter_offset + e];
        uint32_t count = counts_l1[counter_offset + e];
        if (start_page >= max_dispatch_buffer_token_size) {
            count = 0;
        } else if (start_page + count > max_dispatch_buffer_token_size) {
            count = max_dispatch_buffer_token_size - start_page;
        }
        cnt_e[e] = count;
        prefix[e + 1] = prefix[e] + (count * blocks_per_row + tile_h - 1) / tile_h;
    }
    const uint32_t total_blocks = prefix[experts_per_chip];

    // --- This core's contiguous slice of the global block space (must match the reader's). ---
    const uint32_t bpc = (total_blocks + num_cores - 1) / num_cores;
    uint32_t b_lo = core_id * bpc;
    uint32_t b_hi = b_lo + bpc;
    if (b_lo > total_blocks) {
        b_lo = total_blocks;
    }
    if (b_hi > total_blocks) {
        b_hi = total_blocks;
    }

    // --- Write back each global block in [b_lo, b_hi); map it to its (expert, row, block_idx). ---
    uint32_t e = 0;
    while (e < experts_per_chip && prefix[e + 1] <= b_lo) {
        ++e;
    }
    for (uint32_t b = b_lo; b < b_hi; ++b) {
        while (e < experts_per_chip && prefix[e + 1] <= b) {
            ++e;
        }
        const uint32_t start_page = offsets_l1[counter_offset + e];
        const uint32_t end_row = start_page + cnt_e[e];
        const uint32_t total_blocks_e = cnt_e[e] * blocks_per_row;
        const uint32_t start_sb = (b - prefix[e]) * tile_h;
        const uint32_t remaining = total_blocks_e - start_sb;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;

        uint32_t current_row = start_page + start_sb / blocks_per_row;
        uint32_t block_idx_in_row = start_sb % blocks_per_row;

        cb_out_fp32_obj.wait_front(tiles_per_block);
        uint32_t slot = 0;
        while (slot < real_in_block && current_row < end_row) {
            uint32_t blocks_left_in_row = blocks_per_row - block_idx_in_row;
            uint32_t slots_left = real_in_block - slot;
            uint32_t run = blocks_left_in_row < slots_left ? blocks_left_in_row : slots_left;
            noc.async_write(
                cb_out_fp32_obj,
                dst,
                run * out_block_bytes,
                {.offset_bytes = slot * out_block_bytes},
                {.page_id = current_row, .offset_bytes = block_idx_in_row * out_block_bytes});
            slot += run;
            block_idx_in_row += run;
            if (block_idx_in_row >= blocks_per_row) {
                block_idx_in_row = 0;
                ++current_row;
            }
        }
        noc.async_write_barrier();
        cb_out_fp32_obj.pop_front(tiles_per_block);
    }
}
