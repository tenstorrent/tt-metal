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

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

// Compile-time args
constexpr uint32_t cb_input_shard = get_compile_time_arg_val(0);
// Nominal shard width in tiles: the in-shard row stride, also on a short last shard.
constexpr uint32_t shard_tile_cols = get_compile_time_arg_val(1);
constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(2);
constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
constexpr auto output_args = TensorAccessorArgs<4>();

// Per-core RT args: num_blocks, then BLOCK_ARGS words per block.
constexpr uint32_t BLOCK_ARGS = 5;

// Writes rows x cols tiles of the resident shard, starting at in-shard tile local_start, to the
// output pages of the rectangle whose top-left tile is (origin_row, origin_col).
template <typename Accessor>
FORCE_INLINE void store_block(
    const Accessor& output,
    uint32_t shard_base,
    uint32_t origin_row,
    uint32_t origin_col,
    uint32_t rows,
    uint32_t cols,
    uint32_t local_start) {
    uint32_t row_page = origin_row * tensor_tile_cols + origin_col;
    uint32_t row_src = shard_base + local_start * page_bytes;
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t page = row_page;
        uint32_t src = row_src;
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write_page(page, output, src, page_bytes);
            ++page;
            src += page_bytes;
        }
        row_page += tensor_tile_cols;
        row_src += shard_tile_cols * page_bytes;
    }
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(0);

    const auto output = TensorAccessor(output_args, output_addr, page_bytes);
    // No cb_wait_front: no kernel ever pushes to cb_input_shard; the shard is resident at launch.
    const uint32_t shard_base = get_read_ptr(cb_input_shard);

    {
        DeviceZoneScopedN("S2I-STORE-BLOCK");
        for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
            const uint32_t arg = 1 + block_idx * BLOCK_ARGS;
            store_block(
                output,
                shard_base,
                get_arg_val<uint32_t>(arg + 0),
                get_arg_val<uint32_t>(arg + 1),
                get_arg_val<uint32_t>(arg + 2),
                get_arg_val<uint32_t>(arg + 3),
                get_arg_val<uint32_t>(arg + 4));
        }
    }
    {
        // Every write must be acked before return: the next program starts right after.
        DeviceZoneScopedN("S2I-DRAIN-WRITES");
        noc_async_write_barrier();
    }
}
