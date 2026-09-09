// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize writer (NoC1) — `store_block`.
//
// NOT INSTANTIATED on the native sharded-output path: there the packer has
// already written every tile into the output shard's own L1 (cb_output_tiles is
// placed on that buffer), so the program carries no writer kernel at all. This
// file is the store for every destination that is genuinely elsewhere —
// interleaved, DRAM-sharded, or an L1 shard on a DIFFERENT partition from the
// one the blocks were cut on (the cross-spec gather, a real remote write).
//
// RAW-API JUSTIFICATION (the only non-helper block operation in this op).
// The two candidate dataflow helpers were checked against this destination and
// both mismatch concretely:
//   * `write_sticks_after_untilize` is the UNTILIZE counterpart: it writes
//     ROW_MAJOR *sticks*, addressing the destination by stick index
//     (`accessor.get_noc_addr(start_page + block_row + row)` and
//     `l1_addr += padded_row_bytes`, tilize_helpers_dataflow.inl:233-235).
//     This destination is TILE layout, whose pages are whole tiles addressed by
//     TILE index (`row * C + col`), so every address it computes is wrong. Its
//     barrier granularity is also pinned at one tile-row, which is exactly the
//     `write_rows_per_barrier` knob this op has to expose.
//   * `local_copy_helpers_dataflow` is an L1 -> L1 self-aimed-read family whose
//     destination must resolve to AddressType::LOCAL_L1; the output here is an
//     interleaved DRAM tensor.
// Recorded gap: a `write_tile_pages_for_tilize<cb>(accessor, num_tile_rows,
// tiles_per_row, tensor_col_tiles, start_tile_row, col_offset, rows_per_barrier)
// block helper does not exist and would close this — the symmetric counterpart
// to `read_sticks_for_tilize` on the tiled side. This kernel is that block
// operation, written out.
//
// BATCHING. `write_rows_per_barrier` whole-tile-page writes' worth of tile-rows
// go in flight behind ONE barrier, so the transaction count per barrier is
// `write_rows_per_barrier * block_width_tiles` — at or above the measured ~4-8
// in-flight plateau on every block width, including the C == 1 tensor where a
// per-tile-row barrier would put exactly ONE 2 KB write in flight (the trap).
//
// WRAP SAFETY. Neither CB endpoint may wrap mid-transfer: `cb_pop_front` asserts
// `fifo_rd_ptr <= fifo_limit` and only wraps on exact equality (dataflow_api.h:
// 267-272, "consumer always reads from contiguous memory, it cannot wrap"). The
// batch is therefore additionally capped by the pages contiguously readable
// from the current read pointer. With the CB page count an exact multiple of
// `block_width_tiles` and every pop a multiple of it, that cap is always at
// least one tile-row, so the loop always makes progress.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(2);  // R
    constexpr uint32_t tensor_col_tiles = get_compile_time_arg_val(3);   // C
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t write_rows_per_barrier = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    // 1 on the solved plan (contiguous block ranges); the core count on the
    // shard-driven plan, where core i owns shards {i, i+N, i+2N, ...}.
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    // Tile-indexed accessor over the TILE-layout output.
    const auto out_acc = TensorAccessor(out_args, dst_addr);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        // resolve_block — the identical derivation the reader and compute run.
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;
        const uint32_t col_base = w_chunk * block_width_tiles;

        // store_block: `write_rows_per_barrier` tile-rows per barrier.
        uint32_t rows_done = 0;
        while (rows_done < block_row_extent) {
            uint32_t rows_this_batch = block_row_extent - rows_done;
            if (rows_this_batch > write_rows_per_barrier) {
                rows_this_batch = write_rows_per_barrier;
            }
            // Cap by the contiguously readable span (see WRAP SAFETY above).
            {
                const LocalCBInterface& cb = get_local_cb_interface(cb_output_tiles);
                const uint32_t contig_rows = ((cb.fifo_limit - cb.fifo_rd_ptr) / cb.fifo_page_size) / block_width_tiles;
                if (rows_this_batch > contig_rows) {
                    rows_this_batch = contig_rows;
                }
            }

            const uint32_t pages_this_batch = rows_this_batch * block_width_tiles;
            cb_wait_front(cb_output_tiles, pages_this_batch);

            uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
            for (uint32_t r = 0; r < rows_this_batch; ++r) {
                // Output page id of tile (row, col) in an interleaved TILE
                // tensor is row * C + col, with `row` already folding the
                // leading dims (R = num_images * rows_per_image).
                const uint32_t page_base = (row_start + rows_done + r) * tensor_col_tiles + col_base;
                for (uint32_t i = 0; i < block_width_tiles; ++i) {
                    // `out_tile_bytes` as `max_page_size` takes the ONE-PACKET
                    // issue path (dataflow_api.h:838): a whole tile page is
                    // always <= NOC_MAX_BURST_SIZE, so the generic
                    // `*_any_len` multi-packet setup this defaults to is pure
                    // RISC issue cost with nothing to show for it.
                    noc_async_write<out_tile_bytes>(l1_read_addr, out_acc.get_noc_addr(page_base + i), out_tile_bytes);
                    l1_read_addr += out_tile_bytes;
                }
            }

            noc_async_write_barrier();
            cb_pop_front(cb_output_tiles, pages_this_batch);
            rows_done += rows_this_batch;
        }
    }
}
