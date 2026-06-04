// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the e4m3 -> fp32/bf16 pipeline (group-major). The compute produces, per block, a
// [tile_h group-rows x 128] row-major output (COL_BLOCK_TILES tiles). The writer walks the same flat
// group stream as the reader and writes each bank-contiguous run of groups back to its row at column
// gir*128 with one noc_async_write (mirror of the reader's e4m3 reads).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

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

    const uint32_t groups_per_row = width / COL_BLOCK_ELEMS;  // H / 128
    const uint32_t total_groups = num_rows * groups_per_row;
    const uint32_t end_row = start_row + num_rows;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t first_gidx = blk * tile_h;
        const uint32_t remaining = total_groups - first_gidx;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;

        cb_wait_front(cb_out_fp32, COL_BLOCK_TILES);
        uint32_t l1 = get_read_ptr(cb_out_fp32);
        uint32_t g = first_gidx;
        uint32_t slot = 0;
        while (slot < real_in_block && (start_row + g / groups_per_row) < end_row) {
            uint32_t row = start_row + g / groups_per_row;
            uint32_t gir = g % groups_per_row;
            uint32_t run = groups_per_row - gir;
            uint32_t slots_left = real_in_block - slot;
            if (run > slots_left) {
                run = slots_left;
            }
            noc_async_write(
                l1 + slot * out_group_bytes, dst.get_noc_addr(row) + gir * out_group_bytes, run * out_group_bytes);
            slot += run;
            g += run;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_fp32, COL_BLOCK_TILES);
    }
}
