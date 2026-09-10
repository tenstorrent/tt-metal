// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_dual_risc bench — BASELINE writer (BRISC / default NoC1).
//
// Isolated reconstruction of the op's current `store_block`: BRISC alone
// drains cb_output_tiles, `write_rows_per_barrier` tile-rows behind one
// barrier. This is the honest baseline every candidate in this bench is
// measured against.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t write_rows_per_barrier = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    const auto out_acc = TensorAccessor(out_args, dst_addr);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;
        const uint32_t col_base = w_chunk * block_width_tiles;

        uint32_t rows_done = 0;
        while (rows_done < block_row_extent) {
            uint32_t rows_this_batch = block_row_extent - rows_done;
            if (rows_this_batch > write_rows_per_barrier) {
                rows_this_batch = write_rows_per_barrier;
            }
            {
                const LocalCBInterface& cb = get_local_cb_interface(cb_output_tiles);
                const uint32_t contig_rows = ((cb.fifo_limit - cb.fifo_rd_ptr) / cb.fifo_page_size) / block_width_tiles;
                if (rows_this_batch > contig_rows) {
                    rows_this_batch = contig_rows;
                }
            }
            const uint32_t pages_this_batch = rows_this_batch * block_width_tiles;
            {
                MaybeDeviceZoneScope("writer_wait_out");
                cb_wait_front(cb_output_tiles, pages_this_batch);
            }

            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            {
                MaybeDeviceZoneScope("writer_issue");
                for (uint32_t r = 0; r < rows_this_batch; ++r) {
                    const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;
                    for (uint32_t i = 0; i < block_width_tiles; ++i) {
                        noc_async_write<out_tile_bytes>(
                            l1_read_addr, out_acc.get_noc_addr(page_base + i), out_tile_bytes);
                        l1_read_addr += out_tile_bytes;
                    }
                }
            }
            {
                MaybeDeviceZoneScope("writer_barrier");
                noc_async_write_barrier();
            }
            cb_pop_front(cb_output_tiles, pages_this_batch);
            rows_done += rows_this_batch;
        }
    }
}
