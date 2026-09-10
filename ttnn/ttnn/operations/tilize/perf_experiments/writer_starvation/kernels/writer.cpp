// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_starvation bench — writer (BRISC / NoC1).
//
// `wait_tiles == 0` is the op's CURRENT `store_block`, reconstructed from
// `ttnn/ttnn/operations/tilize/kernels/tilize_writer.cpp`: `write_rows_per_
// barrier` tile-rows behind ONE `cb_wait_front` and ONE barrier, with the
// Perf-1 core-dependent issue-order rotation on BOTH axes. That is the honest
// baseline.
//
// `wait_tiles > 0` is the consumer half of IDEA (a): the same batch, the same
// transfers, the same ONE barrier and the same single `cb_pop_front` — but the
// batch's pages are waited for and ISSUED in `wait_tiles`-page groups as the
// packer publishes them, so the first NoC write goes out while compute is still
// on the row's trailing tiles. In-flight depth at the barrier is therefore
// UNCHANGED (the whole batch is still outstanding when the barrier runs); only
// the moment the first transaction is issued moves earlier. Nothing is popped
// before the barrier, so no L1 the NoC is reading can be recycled early.
//
// ROTATION UNDER FINE WAIT. The column rotation is kept verbatim — it permutes
// issue order INSIDE a group, which is free for the baseline's own reason (each
// write's L1 source is index-derived, the whole batch is behind one barrier).
// The ROW rotation is dropped on this path and cannot be otherwise: rows become
// available in ascending order, so starting at row `block_id % rows` would mean
// waiting for the last row before issuing the first — i.e. re-serializing
// exactly what the idea removes. On every geometry in this bench's focus regime
// (`R == 1`, one tile-row per batch) the row rotation is inert anyway; where it
// is not, `baseline_no_row_rot` isolates its cost so the comparison stays
// attributable.
#include "api/dataflow/dataflow_api.h"
#include "bench_zone.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t write_rows_per_barrier = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(7);
    // 0 = the op's current whole-batch wait. > 0 = wait/issue in groups of this
    // many output pages (must divide block_width_tiles).
    constexpr uint32_t wait_tiles = get_compile_time_arg_val(8);
    // Perf-1 ROW rotation on the baseline path. Held as a knob so the fine-wait
    // path's forced ascending row order can be priced separately.
    constexpr uint32_t row_rotate = get_compile_time_arg_val(9);
    constexpr auto out_args = TensorAccessorArgs<10>();

    static_assert(wait_tiles == 0 || block_width_tiles % wait_tiles == 0, "wait_tiles must divide block_width_tiles");

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
        const uint32_t col_rot = block_id % block_width_tiles;

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

            if constexpr (wait_tiles == 0) {
                // ---- the op's current store_block ----------------------------
                {
                    // Named `writer_wait_first` (not `writer_wait_out`) so it
                    // lines up with the fine path's first-group wait: on this
                    // path the batch's first write is issued after exactly this
                    // wait, because there is only one.
                    BenchZone("writer_wait_first");
                    cb_wait_front(cb_output_tiles, pages_this_batch);
                }
                const uint32_t l1_read_base = get_read_ptr(cb_output_tiles);
                const uint32_t row_rot = row_rotate ? (block_id % rows_this_batch) : 0;
                {
                    BenchZone("writer_issue");
                    for (uint32_t rs = 0; rs < rows_this_batch; ++rs) {
                        uint32_t r = rs + row_rot;
                        if (r >= rows_this_batch) {
                            r -= rows_this_batch;
                        }
                        const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;
                        const uint32_t l1_row_addr = l1_read_base + r * block_width_tiles * out_tile_bytes;
                        for (uint32_t cs = 0; cs < block_width_tiles; ++cs) {
                            uint32_t i = cs + col_rot;
                            if (i >= block_width_tiles) {
                                i -= block_width_tiles;
                            }
                            noc_async_write<out_tile_bytes>(
                                l1_row_addr + i * out_tile_bytes, out_acc.get_noc_addr(page_base + i), out_tile_bytes);
                        }
                    }
                }
            } else {
                // ---- IDEA (a) consumer: wait + issue per push group ----------
                const uint32_t l1_read_base = get_read_ptr(cb_output_tiles);
                uint32_t issued = 0;
                while (issued < pages_this_batch) {
                    const uint32_t next = issued + wait_tiles;
                    // THE decisive number for idea (a) is the FIRST group's
                    // wait: that is when the batch's first NoC write is issued,
                    // and it is directly comparable to the baseline's single
                    // `writer_wait_first`. The later groups' waits are a
                    // different quantity (they overlap the NoC drain of the
                    // groups already issued), so they get their own name rather
                    // than being summed into it.
                    if (issued == 0) {
                        BenchZone("writer_wait_first");
                        cb_wait_front(cb_output_tiles, next);
                    } else {
                        BenchZone("writer_wait_rest");
                        // Cumulative: `cb_wait_front(n)` is "at least n pages
                        // at the front", so this is a strictly weaker wait than
                        // the baseline's single whole-batch one.
                        cb_wait_front(cb_output_tiles, next);
                    }
                    {
                        BenchZone("writer_issue");
                        const uint32_t r = issued / block_width_tiles;
                        const uint32_t group_col = issued - r * block_width_tiles;
                        const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;
                        const uint32_t l1_row_addr = l1_read_base + r * block_width_tiles * out_tile_bytes;
                        for (uint32_t cs = 0; cs < wait_tiles; ++cs) {
                            // Rotate WITHIN the group; the group's own tile set
                            // is fixed by `group_col`, so the bytes cannot move.
                            uint32_t i = group_col + ((cs + col_rot) % wait_tiles);
                            noc_async_write<out_tile_bytes>(
                                l1_row_addr + i * out_tile_bytes, out_acc.get_noc_addr(page_base + i), out_tile_bytes);
                        }
                    }
                    issued = next;
                }
            }

            {
                BenchZone("writer_barrier");
                noc_async_write_barrier();
            }
            cb_pop_front(cb_output_tiles, pages_this_batch);
            rows_done += rows_this_batch;
        }
    }
}
