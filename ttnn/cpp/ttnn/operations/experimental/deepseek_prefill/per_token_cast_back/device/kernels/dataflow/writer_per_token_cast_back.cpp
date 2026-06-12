// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the input_e4m3 -> fp32/bf16 pipeline. The compute produces, per block, a
// [tile_h scale-block rows x 128] row-major output (tiles_per_block tiles). The writer walks the
// same flat scale-block stream as the reader and writes each bank-contiguous run back to its row at
// the current block column offset with one NoC async write.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);  // absolute first row of this core's stream
    uint32_t num_rows = get_arg_val<uint32_t>(3);   // rows owned by this core
    uint32_t width = get_arg_val<uint32_t>(4);      // H (elements per row)

    constexpr uint32_t cb_out_fp32 = get_compile_time_arg_val(0);
    constexpr uint32_t out_block_bytes = get_compile_time_arg_val(1);  // 128 * out_elem_size
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(3);  // tiles per block (= 4 for 32-wide tiles)
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const auto dst = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    CircularBuffer cb_out_fp32_obj(cb_out_fp32);

    const uint32_t blocks_per_row = width >> 7;  // H / 128; one-time shift
    const uint32_t total_blocks = num_rows * blocks_per_row;
    const uint32_t end_row = start_row + num_rows;

    // Persistent (row, block_idx) cursor over the flat block stream: no per-run div/mod (expensive on
    // the Baby RISC-V); advance block_idx by the run and reset to the next row with a conditional.
    uint32_t current_row = start_row;
    uint32_t block_idx_in_row = 0;

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
