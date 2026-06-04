// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the input_e4m3 -> fp32/bf16 pipeline. The compute produces, per block, a
// [tile_h scale-block rows x 128] row-major output (COL_BLOCK_TILES tiles). The writer walks the
// same flat scale-block stream as the reader and writes each bank-contiguous run back to its row at
// column gir*128 with one NoC async write.

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
    constexpr uint32_t out_group_bytes = get_compile_time_arg_val(1);  // 128 * out_elem_size (one group)
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t COL_BLOCK_TILES = get_compile_time_arg_val(3);  // tiles per block (= 4)
    constexpr uint32_t COL_BLOCK_ELEMS = 128;                          // column-block width = one group
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const auto dst = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    CircularBuffer cb_out_fp32_obj(cb_out_fp32);

    const uint32_t groups_per_row = width >> 7;  // H / 128 (COL_BLOCK_ELEMS = 128); one-time shift
    const uint32_t total_groups = num_rows * groups_per_row;
    const uint32_t end_row = start_row + num_rows;

    // Persistent (row, gir) cursor over the flat group stream: no per-run div/mod (expensive on the
    // Baby RISC-V); advance gir by the run and reset to the next row with a conditional.
    uint32_t current_row = start_row;
    uint32_t gir = 0;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base = blk * tile_h;
        const uint32_t remaining = total_groups - base;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;

        cb_out_fp32_obj.wait_front(COL_BLOCK_TILES);
        uint32_t slot = 0;
        while (slot < real_in_block && current_row < end_row) {
            uint32_t groups_left_in_row = groups_per_row - gir;
            uint32_t slots_left = real_in_block - slot;
            uint32_t run = groups_left_in_row < slots_left ? groups_left_in_row : slots_left;
            noc.async_write(
                cb_out_fp32_obj,
                dst,
                run * out_group_bytes,
                {.offset_bytes = slot * out_group_bytes},
                {.page_id = current_row, .offset_bytes = gir * out_group_bytes});
            slot += run;
            gir += run;
            if (gir >= groups_per_row) {  // run never crosses a row boundary, so this is exact
                gir = 0;
                ++current_row;
            }
        }
        noc.async_write_barrier();
        cb_out_fp32_obj.pop_front(COL_BLOCK_TILES);
    }
}
