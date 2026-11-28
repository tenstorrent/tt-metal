// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Tile writer (BRISC / NoC1) of the ROW_MAJOR -> TILE regimes: store_tiles_block.
//
// Per block: one cb_wait_front of the nominal block's tile pages (slot i * block_cols_tiles + j,
// produced by the tilize packer), one whole-tile write per live tile of the block, one
// noc_async_writes_flushed, one cb_pop_front. The tile size is the output dtype's (cb_tile_block's
// format), so bf16 and bfp8 tiles share this writer.
// Zones (profiler builds only): per block, the wait and the block's writes each have a zone; the
// flush inside the writes is a sum zone (TT_METAL_PROFILER_SUM).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;

void kernel_main() {
    constexpr uint32_t cb_tile_block = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr auto output_args = TensorAccessorArgs<Geometry::next_ct_arg + 1>();
    constexpr uint32_t block_tiles = Geometry::block_rows_tiles * Geometry::block_cols_tiles;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(cb_tile_block);
    const auto output = TensorAccessor(output_args, output_addr);

    DeviceZoneScopedN("g6_tile_writer");
    CoreBlocks<false> blocks;
    Block block;
    while (blocks.next(block)) {
        {
            DeviceZoneScopedN("wait_tile_block");
            cb_wait_front(cb_tile_block, block_tiles);
        }
        {
            DeviceZoneScopedN("store_tiles_block");
            uint32_t tile_row_src = get_read_ptr(cb_tile_block);
            for (uint32_t i = 0; i < block.rows; ++i) {
                uint32_t page = (block.tile_row + i) * Geometry::tensor_cols_tiles + block.col;
                uint32_t src = tile_row_src;
                for (uint32_t j = 0; j < block.cols; ++j) {
                    noc_async_write(src, output.get_noc_addr(page), tile_bytes);
                    ++page;
                    src += tile_bytes;
                }
                tile_row_src += Geometry::block_cols_tiles * tile_bytes;
            }
            {
                DeviceZoneScopedSumN1("tile_block_writes_flushed");
                noc_async_writes_flushed();
            }
        }
        cb_pop_front(cb_tile_block, block_tiles);
    }
    {
        DeviceZoneScopedN("final_write_barrier");
        noc_async_write_barrier();
    }
}
