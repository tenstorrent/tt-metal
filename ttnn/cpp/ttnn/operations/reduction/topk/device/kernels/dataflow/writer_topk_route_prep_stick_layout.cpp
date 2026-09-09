// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// topk_route_prep writer. Copied-with-attribution from
// ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/
// writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp
// (same structure: wait one untilized CB page, NoC-write its rows onto output
// sticks, barrier, pop). Changes vs the source:
//   1. Row clamp (the design's stated change): the last tile-row of each batch
//      emits only min(32, R - rt*32) logical sticks, so tile-height padding
//      never reaches the ROW_MAJOR output.
//   2. Per-block (rt, wt) derivation + width clamp: the source writer's fixed
//      per-core column offset is only valid for single-tile-row tensors (its
//      parallelize-column factory), while this op splits a general tile grid
//      into per-tile-row blocks — so each block derives its own stick ids and
//      stick offset, and the last block of a row emits only the logical
//      min(bw*32, W - wt0*32) columns (W_p > W tile-width padding dropped).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t nblocks = get_arg_val<uint32_t>(1);
    const uint32_t start_block = get_arg_val<uint32_t>(2);
    const uint32_t nblocks_per_row = get_arg_val<uint32_t>(3);
    const uint32_t tile_rows_per_batch = get_arg_val<uint32_t>(4);  // R_p / 32
    const uint32_t logical_rows = get_arg_val<uint32_t>(5);         // R
    const uint32_t logical_width = get_arg_val<uint32_t>(6);        // W

    constexpr uint32_t bw_full = get_compile_time_arg_val(0);
    constexpr uint32_t bw_last = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t elem_bytes = 2;  // bf16 only (validated host-side)

    // Output pages are W-element logical sticks; the accessor's page size is the
    // compile-time AlignedPageSize baked in by TensorAccessorArgs on the host.
    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_out(cb_out);

    // Blocks are tile-row-major over all batches: block -> (tile_row = b / nblocks_per_row,
    // pos = b % nblocks_per_row); tile_row -> (batch, row_in_batch). Maintained incrementally.
    uint32_t pos = start_block % nblocks_per_row;
    const uint32_t start_tile_row = start_block / nblocks_per_row;
    uint32_t batch = start_tile_row / tile_rows_per_batch;
    uint32_t row_in_batch = start_tile_row % tile_rows_per_batch;

    for (uint32_t b = 0; b < nblocks; ++b) {
        const bool last_in_row = (pos == nblocks_per_row - 1);
        const uint32_t bw = last_in_row ? bw_last : bw_full;

        // Row clamp: emit only the logical sticks of a height-padded last tile-row.
        const uint32_t first_row = row_in_batch * tile_height;
        const uint32_t rows_left = logical_rows - first_row;
        const uint32_t nrows = rows_left < tile_height ? rows_left : tile_height;

        // Width clamp: emit only the logical columns of a width-padded last block.
        const uint32_t col0 = pos * bw_full * tile_width;
        const uint32_t cols_left = logical_width - col0;
        const uint32_t block_cols = bw * tile_width;
        const uint32_t ncols = cols_left < block_cols ? cols_left : block_cols;

        const uint32_t stick_offset_bytes = col0 * elem_bytes;
        const uint32_t write_bytes = ncols * elem_bytes;
        const uint32_t cb_stick_stride = bw * tile_width * elem_bytes;  // untilized block row pitch
        const uint32_t base_stick = batch * logical_rows + first_row;

        dfb_out.wait_front(1);
        uint32_t l1_read_addr = dfb_out.get_read_ptr();
        for (uint32_t k = 0; k < nrows; ++k) {
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(
                src,
                s,
                write_bytes,
                {.offset_bytes = 0},
                {.page_id = base_stick + k, .offset_bytes = stick_offset_bytes});
            l1_read_addr += cb_stick_stride;
        }
        noc.async_write_barrier();
        dfb_out.pop_front(1);

        if (last_in_row) {
            pos = 0;
            if (++row_in_batch == tile_rows_per_batch) {
                row_in_batch = 0;
                ++batch;
            }
        } else {
            ++pos;
        }
    }
}
