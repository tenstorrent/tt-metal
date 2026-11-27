// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding writer: store_block writes a block's whole 2 KB tiles to output pages
// g * tensor_col_tiles + col_block * block_col_tiles + j. Raw NoC writes: kernel_lib has no tile-page
// writer helper (write_sticks_after_untilize writes row-major sticks).
//
// The CB slot is popped only after its writes have left L1 (writes_flushed), and the kernel ends
// with a write barrier so no write is in flight when the next program starts.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
constexpr uint32_t block_col_tiles = get_compile_time_arg_val(1);
constexpr uint32_t block_tile_rows = get_compile_time_arg_val(2);
constexpr uint32_t col_blocks_per_row = get_compile_time_arg_val(3);
constexpr auto output_args = TensorAccessorArgs<4>();

constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;

template <typename Accessor>
FORCE_INLINE void store_block(
    const Accessor& output_accessor,
    uint32_t group,
    uint32_t col_block,
    uint32_t tiles_addr,
    uint32_t tile_bytes,
    uint32_t tensor_tile_rows,
    uint32_t tensor_col_tiles) {
    for (uint32_t t = 0; t < block_tile_rows; ++t) {
        const uint32_t g = group * block_tile_rows + t;
        if (g >= tensor_tile_rows) {
            break;
        }
        const uint32_t first_page = g * tensor_col_tiles + col_block * block_col_tiles;
        for (uint32_t j = 0; j < block_col_tiles; ++j) {
            noc_async_write(tiles_addr, output_accessor.get_noc_addr(first_page + j), tile_bytes);
            tiles_addr += tile_bytes;
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tensor_tile_rows = get_common_arg_val<uint32_t>(1);
    const uint32_t tensor_col_tiles = get_common_arg_val<uint32_t>(2);
    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(1);

    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const uint32_t tile_bytes = get_tile_size(cb_output_tiles);

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t block_id = first_block + block_idx;
        const uint32_t group = block_id / col_blocks_per_row;
        const uint32_t col_block = block_id - group * col_blocks_per_row;

        {
            DeviceZoneScopedN("out_wait");
            cb_wait_front(cb_output_tiles, block_pages);
        }
        {
            DeviceZoneScopedN("store_block");
            store_block(
                output_accessor,
                group,
                col_block,
                get_read_ptr(cb_output_tiles),
                tile_bytes,
                tensor_tile_rows,
                tensor_col_tiles);
            noc_async_writes_flushed();
            cb_pop_front(cb_output_tiles, block_pages);
        }
    }
    noc_async_write_barrier();
}
