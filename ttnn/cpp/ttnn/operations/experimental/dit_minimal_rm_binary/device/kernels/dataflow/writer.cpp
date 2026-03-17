// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writes row-major sticks from the untilize output CB back to DRAM.
//
// untilize_block produces ntiles_per_row tile-sized (2048-byte) pages per
// row-block, with data laid out row-major within the block:
//   row k starts at byte offset k * stick_size from the block base.
//
// For each row-block the writer pops the whole ntiles_per_row-page block at
// once and issues TILE_HEIGHT NOC writes (one per row) at the computed offsets.
//
// The last block may be partial (num_sticks % TILE_HEIGHT != 0).  In that case
// only the first tail_rows rows carry real data; the remainder is padding that
// is never written to DRAM.
//
// Stick (DRAM page) == one full tensor row == stick_size bytes.
//
// CT args: [cb_out, stick_size, TensorAccessorArgs_out...,
//           ntiles_per_row, tile_width_bytes]
//   cb_out           = output CB index
//   stick_size       = row_size_bytes (= ntiles_per_row * tile_width_bytes)
//   ntiles_per_row   = number of tile-columns (pages per CB block)
//   tile_width_bytes = TILE_WIDTH * dtype_bytes (unused, kept for CT-arg layout
//                      compatibility with the reader)
//
// RT args: [dst_addr, num_sticks, start_stick_id]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(out_args.next_compile_time_args_offset());
    constexpr uint32_t tile_width_bytes = get_compile_time_arg_val(out_args.next_compile_time_args_offset() + 1);

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    const auto dst = TensorAccessor(out_args, dst_addr, stick_size);

    uint32_t stick_id = start_stick_id;

    const uint32_t num_full_blocks = num_sticks / TILE_HEIGHT;
    const uint32_t tail_rows = num_sticks % TILE_HEIGHT;

    // Full blocks: pop ntiles_per_row pages, write TILE_HEIGHT rows.
    // Row k is at base_l1 + k * stick_size.
    for (uint32_t i = 0; i < num_full_blocks; ++i) {
        cb_wait_front(cb_out_id, ntiles_per_row);
        uint32_t base_l1 = get_read_ptr(cb_out_id);
        for (uint32_t k = 0; k < TILE_HEIGHT; ++k) {
            noc_async_write(base_l1 + k * stick_size, dst.get_noc_addr(stick_id), stick_size);
            ++stick_id;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_id, ntiles_per_row);
    }

    // Partial last block: pop ntiles_per_row pages, write only tail_rows rows.
    if (tail_rows > 0) {
        cb_wait_front(cb_out_id, ntiles_per_row);
        uint32_t base_l1 = get_read_ptr(cb_out_id);
        for (uint32_t k = 0; k < tail_rows; ++k) {
            noc_async_write(base_l1 + k * stick_size, dst.get_noc_addr(stick_id), stick_size);
            ++stick_id;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_id, ntiles_per_row);
    }
}
