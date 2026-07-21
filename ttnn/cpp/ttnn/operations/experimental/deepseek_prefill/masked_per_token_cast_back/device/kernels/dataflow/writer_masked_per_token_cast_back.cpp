// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for masked_per_token_cast_back. Same column-block-major write logic as per_token_cast_back,
// but the row range is derived on-device: this kernel reads the per-expert region offsets / token
// counts / global-expert-idx table, computes total_valid_rows, and takes the same balanced tile-row
// slice as the reader (identical (core_id, num_cores) formula), so its output CB is drained by exactly
// its own writer. Only the valid prefix rows are written; the garbage tail is left untouched.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/assert.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t region_addr = get_arg_val<uint32_t>(1);
    uint32_t counts_addr = get_arg_val<uint32_t>(2);
    uint32_t table_addr = get_arg_val<uint32_t>(3);
    uint32_t core_id = get_arg_val<uint32_t>(4);
    uint32_t width = get_arg_val<uint32_t>(5);  // H (elements per row)

    constexpr uint32_t cb_out_fp32 = get_compile_time_arg_val(0);
    constexpr uint32_t out_block_bytes = get_compile_time_arg_val(1);  // 128 * out_elem_size
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(3);  // tiles per block (= 4 for 32-wide tiles)
    constexpr uint32_t cb_region_scratch = get_compile_time_arg_val(4);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(5);
    constexpr uint32_t cb_table_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(8);
    // Bounds for validating untrusted device metadata (watcher-build asserts below).
    constexpr uint32_t num_routed_experts = get_compile_time_arg_val(9);  // length of region/counts vectors
    constexpr uint32_t input_num_rows = get_compile_time_arg_val(10);     // M: output buffer row capacity

    constexpr auto dst_args = TensorAccessorArgs<11>();
    constexpr auto region_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<region_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();

    const auto dst = TensorAccessor(dst_args, dst_addr);
    const auto region = TensorAccessor(region_args, region_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);
    const auto table = TensorAccessor(table_args, table_addr);
    Noc noc;
    CircularBuffer cb_out_fp32_obj(cb_out_fp32);
    CircularBuffer cb_region_scratch_obj(cb_region_scratch);
    CircularBuffer cb_counts_scratch_obj(cb_counts_scratch);
    CircularBuffer cb_table_scratch_obj(cb_table_scratch);

    // --- Read the three metadata vectors (small, 1 page each) into private L1 scratch. ---
    cb_region_scratch_obj.reserve_back(1);
    cb_counts_scratch_obj.reserve_back(1);
    cb_table_scratch_obj.reserve_back(1);
    noc.async_read(region, cb_region_scratch_obj, region.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(counts, cb_counts_scratch_obj, counts.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(table, cb_table_scratch_obj, table.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    const volatile tt_l1_ptr uint32_t* region_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_region_scratch_obj.get_write_ptr());
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_counts_scratch_obj.get_write_ptr());
    const volatile tt_l1_ptr uint32_t* table_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_table_scratch_obj.get_write_ptr());

    // Valid prefix length = max over local experts of (region_offset[g] + ceil_tile(counts[g])).
    // Device metadata is untrusted: the asserts (watcher build only) catch a stale/garbage table entry
    // before it indexes past the region/counts scratch pages or drives a NoC write past the output buffer.
    uint32_t total_valid_rows = 0;
    for (uint32_t local_slot = 0; local_slot < experts_per_chip; ++local_slot) {
        const uint32_t global_expert_id = table_ptr[local_slot];
        ASSERT(global_expert_id < num_routed_experts);
        const uint32_t token_count = counts_ptr[global_expert_id];
        const uint32_t token_count_ceil = ((token_count + tile_h - 1) / tile_h) * tile_h;
        const uint32_t region_end = region_ptr[global_expert_id] + token_count_ceil;
        if (region_end > total_valid_rows) {
            total_valid_rows = region_end;
        }
    }
    ASSERT(total_valid_rows <= input_num_rows);

    // Balanced split over the FLATTENED compute-block space (identical formula to the reader), so the
    // reader/compute/writer on the same core agree on the exact (start_row, start_col) .. flattened range.
    const uint32_t blocks_per_row = width >> 7;  // H / 128; one-time shift
    const uint32_t total_tile_rows = total_valid_rows / tile_h;
    const uint32_t total_compute_blocks = total_tile_rows * blocks_per_row;
    const uint32_t cb_start = (total_compute_blocks * core_id) / num_cores;
    const uint32_t cb_end = (total_compute_blocks * (core_id + 1)) / num_cores;
    const uint32_t num_blocks = cb_end - cb_start;
    // Flattened scale-block indices (tile_h scale-blocks per compute-block). One-time div/mod only.
    const uint32_t start_flat = cb_start * tile_h;
    const uint32_t end_flat = cb_end * tile_h;  // exclusive
    const uint32_t start_row = start_flat / blocks_per_row;
    const uint32_t start_col = start_flat % blocks_per_row;
    const uint32_t end_row = (end_flat == 0) ? 0 : ((end_flat - 1) / blocks_per_row) + 1;
    const uint32_t total_blocks = num_blocks * tile_h;  // per-core scale-blocks (exact multiple of tile_h)

    // Persistent (row, block_idx) cursor over the flat block stream: no per-run div/mod (expensive on
    // the Baby RISC-V); advance block_idx by the run and reset to the next row with a conditional.
    uint32_t current_row = start_row;
    uint32_t block_idx_in_row = start_col;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base = blk * tile_h;
        const uint32_t remaining = total_blocks - base;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;

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
            if (block_idx_in_row >= blocks_per_row) {  // run never crosses a row boundary, so this is exact
                block_idx_in_row = 0;
                ++current_row;
            }
        }
        noc.async_write_barrier();
        cb_out_fp32_obj.pop_front(tiles_per_block);
    }
}
