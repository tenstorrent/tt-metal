// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader — NCRISC / NoC0.
//
// One block = one output tile-row x `w` output tile-columns. Blocks are
// linearized `b = wchunk * nt_h + r`; this core owns the contiguous range
// [b0, b0 + nb). Per block the reader issues `tile_h` NoC reads of
// `w * 32 * elem` bytes at L1 stride `w * 32 * elem` and ONE barrier, then
// pushes `w` tile-sized pages — the read/push contract
// `dataflow_kernel_lib::read_sticks_for_tilize<TILE>` implements.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t nt_h = get_compile_time_arg_val(1);             // tile-rows
    constexpr uint32_t n_wchunks = get_compile_time_arg_val(2);        // column-blocks per tile-row
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);           // sticks per tile-row
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(4);  // WT_BLOCK * 32 * elem
    constexpr uint32_t tail_row_bytes = get_compile_time_arg_val(5);   // WT_TAIL  * 32 * elem
    constexpr auto src_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t b0 = get_arg_val<uint32_t>(1);
    const uint32_t nb = get_arg_val<uint32_t>(2);

    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t i = 0; i < nb; ++i) {
        const uint32_t b = b0 + i;
        const uint32_t wchunk = b / nt_h;      // column-block index
        const uint32_t r = b - wchunk * nt_h;  // global tile-row index

        // Tail column-block is the last one; its width is WT_TAIL (== WT_BLOCK
        // when Wt divides evenly), so the reader's per-block page count matches
        // compute's `WT_BLOCK x n_full` then `WT_TAIL x n_tail` sequence exactly.
        const uint32_t row_bytes = (wchunk == n_wchunks - 1) ? tail_row_bytes : chunk_row_bytes;

        dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
            src,
            /*total_num_rows*/ tile_h,
            /*row_bytes*/ row_bytes,
            /*start_page*/ r * tile_h,
            /*byte_offset_within_page*/ wchunk * chunk_row_bytes);
    }
}
