// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_dual_risc bench — candidate A, NCRISC's kernel (default NoC0).
//
// ONE kernel per RISC-V, so the same file that reads the block ALSO carries
// this core's write share — exactly the shape the op's own SPLIT READER uses
// on tilize_writer.cpp (that one adds a trailing READ to the writer; this one
// adds a trailing WRITE to the reader). It:
//   1. reads the block's LEFT half-width columns (all `block_row_extent`
//      rows), then the RIGHT half-width columns (all rows) — TWO
//      `read_sticks_for_tilize` calls into the SAME cb_input_rows, using the
//      helper's documented wide-W chunking parameter
//      (`byte_offset_within_page`) to select each half. One producer
//      (this kernel), one consumer (compute_col_split.cpp), FIFO order
//      preserved.
//   2. once this core's OWN reads are done, drains cb_output_tiles_split (the
//      RIGHT half, produced by compute's SECOND tilize call) and stores it —
//      the mirror of tilize_writer.cpp's `store_block`, restricted to the
//      RIGHT half-width columns.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles_split = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_h = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(5);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t write_rows_per_barrier = get_compile_time_arg_val(8);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(10);  // FULL width, in bytes
    constexpr uint32_t half_width_tiles = block_width_tiles / 2;
    constexpr uint32_t half_row_bytes = block_row_bytes / 2;
    constexpr auto in_args = TensorAccessorArgs<11>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_block_id = get_arg_val<uint32_t>(2);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(3);
    const uint32_t block_stride = get_arg_val<uint32_t>(4);

    const auto in_acc = TensorAccessor(in_args, src_addr);
    const auto out_acc = TensorAccessor(out_args, dst_addr);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;
        const uint32_t block_byte_base = w_chunk * block_row_bytes;

        {
            MaybeDeviceZoneScope("reader_col_split_read_left");
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rows, dataflow_kernel_lib::TilizeGranularity::TILE>(
                in_acc,
                /* total_num_rows          */ block_row_extent * tile_h,
                /* row_bytes               */ half_row_bytes,
                /* start_page              */ row_start * tile_h,
                /* byte_offset_within_page */ block_byte_base);
        }
        {
            MaybeDeviceZoneScope("reader_col_split_read_right");
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rows, dataflow_kernel_lib::TilizeGranularity::TILE>(
                in_acc,
                /* total_num_rows          */ block_row_extent * tile_h,
                /* row_bytes               */ half_row_bytes,
                /* start_page              */ row_start * tile_h,
                /* byte_offset_within_page */ block_byte_base + half_row_bytes);
        }

        // --- write share: this block's RIGHT half-width columns ------------
        const uint32_t col_base = w_chunk * block_width_tiles + half_width_tiles;
        uint32_t rows_done = 0;
        while (rows_done < block_row_extent) {
            uint32_t rows_this_batch = block_row_extent - rows_done;
            if (rows_this_batch > write_rows_per_barrier) {
                rows_this_batch = write_rows_per_barrier;
            }
            {
                const LocalCBInterface& cb = get_local_cb_interface(cb_output_tiles_split);
                const uint32_t contig_rows = ((cb.fifo_limit - cb.fifo_rd_ptr) / cb.fifo_page_size) / half_width_tiles;
                if (rows_this_batch > contig_rows) {
                    rows_this_batch = contig_rows;
                }
            }
            const uint32_t pages_this_batch = rows_this_batch * half_width_tiles;
            {
                MaybeDeviceZoneScope("writer_share_wait_out");
                cb_wait_front(cb_output_tiles_split, pages_this_batch);
            }

            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles_split);
            {
                MaybeDeviceZoneScope("writer_share_issue");
                for (uint32_t r = 0; r < rows_this_batch; ++r) {
                    const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;
                    for (uint32_t i = 0; i < half_width_tiles; ++i) {
                        noc_async_write<out_tile_bytes>(
                            l1_read_addr, out_acc.get_noc_addr(page_base + i), out_tile_bytes);
                        l1_read_addr += out_tile_bytes;
                    }
                }
            }
            {
                MaybeDeviceZoneScope("writer_share_barrier");
                noc_async_write_barrier();
            }
            cb_pop_front(cb_output_tiles_split, pages_this_batch);
            rows_done += rows_this_batch;
        }
    }
}
