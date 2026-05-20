// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for layer_norm_rm (Refinement 2 — streaming + multi-core).
//
// Per-core slice = `Ht_local` consecutive tile-rows starting at
// `start_tile_row`.  Per tile-row, drains NUM_BLOCKS blocks of BLOCK_SIZE
// tiles each from cb_output to DRAM.
//
//   TILE output: per block: read BLOCK_SIZE tiles, noc_async_write_tile each
//                to (global_tile_row * Wt + b * BLOCK_SIZE + wt).
//   RM output:  per block: cb_output holds BLOCK_SIZE tile-sized pages laid
//                out by untilize as 32 rows × `block_row_bytes` bytes. Write
//                32 partial-row sticks of `bytes_this_block` bytes to DRAM at
//                offset `b * block_row_bytes` within each row.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(2);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t padded_row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t output_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t last_block_output_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t elem_size = get_compile_time_arg_val(7);
    constexpr uint32_t has_partial_w = get_compile_time_arg_val(8);
    constexpr uint32_t is_rm_output = get_compile_time_arg_val(9);

    constexpr auto dst_args = TensorAccessorArgs<10>();

    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    uint32_t Ht_local = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output = 16;
    constexpr uint32_t TILE_H = 32;

    if (Ht_local == 0) {
        return;
    }

    constexpr uint32_t non_last_block_bytes = block_row_bytes;

    if constexpr (is_rm_output) {
        // RM output: 32 partial-row writes per block × NUM_BLOCKS blocks × Ht_local rows.
        const auto accessor = TensorAccessor(dst_args, output_addr, padded_row_bytes);
        for (uint32_t tr = 0; tr < Ht_local; ++tr) {
            uint32_t global_tile_row = start_tile_row + tr;
            uint32_t base_row = global_tile_row * TILE_H;

            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                uint32_t bytes_this_block = (b + 1 == NUM_BLOCKS) ? last_block_output_bytes : non_last_block_bytes;
                uint32_t col_offset = b * block_row_bytes;

                cb_wait_front(cb_output, BLOCK_SIZE);
                uint32_t l1_base = get_read_ptr(cb_output);
                // L1 layout after untilize<BLOCK_SIZE, …>(1): 32 contiguous
                // rows of `block_row_bytes` bytes each.
                for (uint32_t r = 0; r < TILE_H; ++r) {
                    uint32_t page_id = base_row + r;
                    uint64_t noc_addr = accessor.get_noc_addr(page_id) + col_offset;
                    uint32_t l1_addr = l1_base + r * block_row_bytes;
                    noc_async_write(l1_addr, noc_addr, bytes_this_block);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output, BLOCK_SIZE);
            }
        }
    } else {
        // TILE output: drain BLOCK_SIZE tiles per block × NUM_BLOCKS blocks × Ht_local rows.
        constexpr uint32_t tile_bytes_v = get_tile_size(cb_output);
        const auto accessor = TensorAccessor(dst_args, output_addr, tile_bytes_v);
        for (uint32_t tr = 0; tr < Ht_local; ++tr) {
            uint32_t global_tile_row = start_tile_row + tr;
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                for (uint32_t wt = 0; wt < BLOCK_SIZE; ++wt) {
                    uint32_t tile_id = global_tile_row * Wt + b * BLOCK_SIZE + wt;
                    cb_wait_front(cb_output, 1);
                    uint32_t l1_addr = get_read_ptr(cb_output);
                    noc_async_write_tile(tile_id, accessor, l1_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_output, 1);
                }
            }
        }
    }
}
