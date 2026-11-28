// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf-1 experiment tensix_untilize_bf16: tile reader (NCRISC / NoC0) of the bf16 packer path.
//
// R3's load_block for a bf16 input, without NaN marks: per block, whole-tile reads of the block's
// live tiles at slot i * block_cols_tiles + j of the nominal block (the order the untilize consumes).
// read_ahead blocks are on the wire at once, each under its own trid, in a ring of
// tile_window_blocks nominal blocks addressed from the CB base (read_ahead = 1 is R3's
// reserve / read / barrier / push). A block is pushed once its own trid has landed.
// live_reads = 1: only each tile's live face-row prefixes are read (R1's read_live_tile; adjacent
// whole faces merged, so an interior tile is still one 2048-B read). The rest of the slot is stale:
// the packer untilizes it into stage bytes the writer never writes (rows >= live_rows, bytes past 2W).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_to_layout_block_walk.hpp"

using namespace g6_to_layout;
namespace dkl = dataflow_kernel_lib;

namespace {

// R1's read_live_tile (g6_to_layout_staged_reader.cpp): live face-row prefixes of one tile into a
// 2048-B slot at the same face offsets; adjacent whole faces merged into one read.
FORCE_INLINE void read_live_tile(uint64_t src, uint32_t dst, uint32_t live_rows, uint32_t live_cols) {
    const uint32_t top_rows = min_u32(live_rows, FACE_HW);
    const uint32_t bottom_rows = live_rows - top_rows;
    if (live_cols > FACE_HW) {
        if (bottom_rows == FACE_HW) {
            noc_async_read(src, dst, TILE_BYTES);
        } else if (bottom_rows > 0) {
            noc_async_read(src, dst, 2 * FACE_BYTES + bottom_rows * FACE_ROW_BYTES);
            noc_async_read(src + 3 * FACE_BYTES, dst + 3 * FACE_BYTES, bottom_rows * FACE_ROW_BYTES);
        } else if (top_rows == FACE_HW) {
            noc_async_read(src, dst, 2 * FACE_BYTES);
        } else {
            noc_async_read(src, dst, top_rows * FACE_ROW_BYTES);
            noc_async_read(src + FACE_BYTES, dst + FACE_BYTES, top_rows * FACE_ROW_BYTES);
        }
    } else {
        noc_async_read(src, dst, top_rows * FACE_ROW_BYTES);
        if (bottom_rows > 0) {
            noc_async_read(src + 2 * FACE_BYTES, dst + 2 * FACE_BYTES, bottom_rows * FACE_ROW_BYTES);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_tile_window = get_compile_time_arg_val(Geometry::next_ct_arg + 0);
    constexpr uint32_t read_ahead = get_compile_time_arg_val(Geometry::next_ct_arg + 1);
    constexpr uint32_t tile_window_blocks = get_compile_time_arg_val(Geometry::next_ct_arg + 2);
    constexpr bool live_reads = get_compile_time_arg_val(Geometry::next_ct_arg + 3) != 0;
    constexpr auto input_args = TensorAccessorArgs<Geometry::next_ct_arg + 4>();
    static_assert(read_ahead >= 1 && read_ahead <= tile_window_blocks && read_ahead <= 15);

    constexpr uint32_t block_pages = Geometry::block_rows_tiles * Geometry::block_cols_tiles;
    constexpr uint32_t block_bytes = block_pages * TILE_BYTES;

    DeviceZoneScopedN("g6tu_tile_reader");
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const auto input = TensorAccessor(input_args, input_addr, TILE_BYTES);
    const uint32_t window_base = get_write_ptr(cb_tile_window);
    Noc noc;

    CoreBlocks<false> blocks;
    Block block;
    uint32_t issued = 0;
    uint32_t pushed = 0;

    auto issue = [&]() -> bool {
        if (!blocks.next(block)) {
            return false;
        }
        const uint32_t slot = issued % tile_window_blocks;
        dkl::set_read_trid(noc, 1 + (issued % read_ahead));
        uint32_t row_slot = window_base + slot * block_bytes;
        for (uint32_t i = 0; i < block.rows; ++i) {
            uint32_t page = (block.tile_row + i) * Geometry::tensor_cols_tiles + block.col;
            uint32_t dst = row_slot;
            const uint32_t live_rows = live_rows_of(block.tile_row + i);
            for (uint32_t j = 0; j < block.cols; ++j) {
                if constexpr (live_reads) {
                    read_live_tile(input.get_noc_addr(page), dst, live_rows, live_cols_of(block.col + j));
                } else {
                    noc_async_read(input.get_noc_addr(page), dst, TILE_BYTES);
                }
                ++page;
                dst += TILE_BYTES;
            }
            row_slot += Geometry::block_cols_tiles * TILE_BYTES;
        }
        ++issued;
        return true;
    };

    bool more = true;
    for (;;) {
        // Keep read_ahead blocks on the wire: reserve the window pages of every block issued but not
        // yet pushed plus the next one, then issue it.
        while (more && issued - pushed < read_ahead) {
            {
                DeviceZoneScopedN("reserve_tile_window");
                cb_reserve_back(cb_tile_window, (issued - pushed + 1) * block_pages);
            }
            more = issue();
        }
        if (pushed == issued) {
            break;
        }
        {
            DeviceZoneScopedSumN1("tile_window_read_barrier");
            dkl::async_read_barrier_with_trid(noc, 1 + (pushed % read_ahead));
        }
        cb_push_back(cb_tile_window, block_pages);
        ++pushed;
    }
    dkl::set_read_trid(noc, 0);
}
