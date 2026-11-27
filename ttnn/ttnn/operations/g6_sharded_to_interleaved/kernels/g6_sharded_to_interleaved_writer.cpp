// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Owner-writes-whole-shard: every core writes the tiles of its own resident shard to their
// interleaved output pages. The shard is read in place through cb_input_shard (backed on the
// sharded input buffer); each tile crosses the NoC exactly once, as the write to its page.
//
// Raw NoC writes, no kernel_lib helper: no dataflow helper writes a resident L1 rectangle of pages
// to an interleaved TensorAccessor (local_copy_helpers_dataflow.hpp only reads into local L1,
// McastArgs fans one source out to a receiver group). One write per page is the transaction
// minimum: consecutive output pages round-robin across banks, so no write can cover two pages.
//
// Each core derives its block from its own logical coordinate and the compile-time shard
// geometry, so the program carries no per-core runtime args: the dispatcher writes those core by
// core, which costs more per call under trace replay than the whole block of writes. The host
// plan (describe_plan) mirrors this derivation.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_input_shard = get_compile_time_arg_val(0);
constexpr uint32_t shard_tile_rows = get_compile_time_arg_val(1);
// Nominal shard width in tiles: the in-shard row stride, also on a short last shard.
constexpr uint32_t shard_tile_cols = get_compile_time_arg_val(2);
constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(3);
constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(4);
constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
constexpr uint32_t num_shard_cols = get_compile_time_arg_val(6);
constexpr uint32_t num_shards = get_compile_time_arg_val(7);
constexpr bool walk_row_major = get_compile_time_arg_val(8) != 0;
constexpr uint32_t num_writer_riscs = get_compile_time_arg_val(9);
constexpr uint32_t writer_riscv = get_compile_time_arg_val(10);
constexpr bool rotate_write_order = get_compile_time_arg_val(11) != 0;
constexpr uint32_t num_ranges = get_compile_time_arg_val(12);
// Shard grid ranges in stored order, (x0, y0, x1, y1) each.
constexpr uint32_t ranges_arg = 13;
constexpr auto output_args = TensorAccessorArgs<ranges_arg + 4 * num_ranges>();

// Index of this core in the shard walk: the ranges in stored order, each walked along its rows
// (row-major orientation) or its columns, as the sharded buffer places shard k on the k-th core.
FORCE_INLINE uint32_t shard_walk_index() {
    const uint32_t x = get_absolute_logical_x();
    const uint32_t y = get_absolute_logical_y();
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_ranges; ++i) {
        const uint32_t x0 = kernel_compile_time_args[ranges_arg + 4 * i + 0];
        const uint32_t y0 = kernel_compile_time_args[ranges_arg + 4 * i + 1];
        const uint32_t x1 = kernel_compile_time_args[ranges_arg + 4 * i + 2];
        const uint32_t y1 = kernel_compile_time_args[ranges_arg + 4 * i + 3];
        const uint32_t width = x1 - x0 + 1;
        const uint32_t height = y1 - y0 + 1;
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
            return offset + (walk_row_major ? (y - y0) * width + (x - x0) : (x - x0) * height + (y - y0));
        }
        offset += width * height;
    }
    return offset;  // outside the grid: past every shard
}

struct Block {
    uint32_t origin_row;  // first tensor tile row
    uint32_t origin_col;  // first tensor tile column
    uint32_t rows;
    uint32_t cols;
    uint32_t local_start;  // in-shard tile index of the top-left tile
    uint32_t start_row;    // block tile the write walk begins at
    uint32_t start_col;
};

// This RISC's block of shard k: the shard's actual extent (a last shard may be short), cut into
// num_writer_riscs near-equal pieces along its longer axis. Returns the number of blocks (0 or 1).
FORCE_INLINE uint32_t riscv_block(uint32_t k, Block& b) {
    if (k >= num_shards) {
        return 0;
    }
    b.origin_row = (k / num_shard_cols) * shard_tile_rows;
    b.origin_col = (k % num_shard_cols) * shard_tile_cols;
    b.rows = tensor_tile_rows - b.origin_row < shard_tile_rows ? tensor_tile_rows - b.origin_row : shard_tile_rows;
    b.cols = tensor_tile_cols - b.origin_col < shard_tile_cols ? tensor_tile_cols - b.origin_col : shard_tile_cols;

    const bool cut_rows = b.rows >= b.cols;
    const uint32_t extent = cut_rows ? b.rows : b.cols;
    const uint32_t base = extent / num_writer_riscs;
    const uint32_t rem = extent % num_writer_riscs;
    const uint32_t length = base + (writer_riscv < rem ? 1 : 0);
    const uint32_t offset = writer_riscv * base + (writer_riscv < rem ? writer_riscv : rem);
    if (length == 0) {
        return 0;
    }
    if (cut_rows) {
        b.origin_row += offset;
        b.rows = length;
        b.local_start = offset * shard_tile_cols;
    } else {
        b.origin_col += offset;
        b.cols = length;
        b.local_start = offset;
    }

    b.start_row = 0;
    b.start_col = 0;
    if constexpr (rotate_write_order) {
        const uint32_t start = k % (b.rows * b.cols);
        b.start_row = start / b.cols;
        b.start_col = start % b.cols;
    }
    return 1;
}

// Writes every tile of the block to its output page. The walk is row-major over the block,
// rotated to begin at (start_row, start_col): cores that start at different offsets spread
// their simultaneous writes over different banks.
template <typename Accessor>
FORCE_INLINE void store_block(const Accessor& output, uint32_t shard_base, const Block& b) {
    constexpr uint32_t shard_row_bytes = shard_tile_cols * page_bytes;
    const uint32_t first_row_page = b.origin_row * tensor_tile_cols + b.origin_col;
    const uint32_t first_row_src = shard_base + b.local_start * page_bytes;

    uint32_t row_page = first_row_page + b.start_row * tensor_tile_cols;
    uint32_t row_src = first_row_src + b.start_row * shard_row_bytes;
    uint32_t r = b.start_row;
    uint32_t c = b.start_col;
    for (uint32_t i = b.rows * b.cols; i > 0; --i) {
        noc_async_write_page(row_page + c, output, row_src + c * page_bytes, page_bytes);
        if (++c == b.cols) {
            c = 0;
            if (++r == b.rows) {
                r = 0;
                row_page = first_row_page;
                row_src = first_row_src;
            } else {
                row_page += tensor_tile_cols;
                row_src += shard_row_bytes;
            }
        }
    }
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    // No cb_wait_front: no kernel ever pushes to cb_input_shard; the shard is resident at launch.
    const uint32_t shard_base = get_read_ptr(cb_input_shard);

    Block block;
    const uint32_t num_blocks_this_core = riscv_block(shard_walk_index(), block);
    {
        DeviceZoneScopedN("S2I-STORE-BLOCK");
        for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
            store_block(output, shard_base, block);
        }
    }
    {
        // Every write must be acked before return: the next program starts right after.
        DeviceZoneScopedN("S2I-DRAIN-WRITES");
        noc_async_write_barrier();
    }
}
