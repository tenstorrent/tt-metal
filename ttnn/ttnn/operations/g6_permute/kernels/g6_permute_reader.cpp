// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute reader (rows_tile): load_rows_block.
// Per block, the live input tiles (n, a, b0 div 32, tw), a in the block's a-tile, each get one
// tile-page slot in cb_input_rows; only the block's rows are read, each landing at its own
// intra-tile offset (Blackhole DRAM reads need local and remote address equal mod 64).
// Raw dataflow: no kernel_lib helper reads row ranges of a DRAM tile.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"

using namespace g6_permute;

constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
constexpr uint32_t A = get_compile_time_arg_val(1);
constexpr uint32_t B = get_compile_time_arg_val(2);
constexpr uint32_t tensor_a_tiles = get_compile_time_arg_val(3);
constexpr uint32_t tensor_b_tiles = get_compile_time_arg_val(4);
constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(5);
constexpr uint32_t a_tile_rows = get_compile_time_arg_val(6);
constexpr uint32_t block_b_rows = get_compile_time_arg_val(7);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(8);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(9);
constexpr auto input_args = TensorAccessorArgs<10>();

using Geometry = TileGeometry<face_row_bytes>;
static_assert(Geometry::tile_bytes == tile_bytes, "tile page must be four 16x16 faces");

using Cursor = BlockCursor<B, tensor_a_tiles, tensor_w_tiles, block_b_rows>;

template <typename Accessor>
FORCE_INLINE void issue_row_reads(const Accessor& input, const Cursor& cursor, uint32_t run) {
    const uint32_t slot_base = get_write_ptr(cb_input_rows);
    const uint32_t r0 = cursor.b % TILE_ROWS;
    ByteRange ranges[4];
    const uint32_t num_ranges = row_span_ranges<face_row_bytes>(r0, r0 + run, ranges);

    const uint32_t a_live = cursor.a_live(A);
    const uint32_t a0 = cursor.ta * TILE_ROWS;
    const uint32_t tb = cursor.b / TILE_ROWS;
    // Concurrent cores own consecutive b of the same input tiles; rotating the a-walk start
    // by b spreads their first reads over different tiles (banks).
    uint32_t slot = cursor.b % a_live;
    for (uint32_t i = 0; i < a_live; ++i) {
        const uint32_t page = ((cursor.n * A + a0 + slot) * tensor_b_tiles + tb) * tensor_w_tiles + cursor.tw;
        const uint32_t slot_addr = slot_base + slot * tile_bytes;
        for (uint32_t k = 0; k < num_ranges; ++k) {
            noc_async_read(input.get_noc_addr(page, ranges[k].offset), slot_addr + ranges[k].offset, ranges[k].len);
        }
        if (++slot == a_live) {
            slot = 0;
        }
    }
}

template <bool zoned, typename Accessor>
FORCE_INLINE void load_rows_block(const Accessor& input, const Cursor& cursor, uint32_t run) {
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_rows, a_tile_rows));
    G6_ZONED_STEP(zoned, "PERM-RD-ISSUE", issue_row_reads(input, cursor, run));
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", noc_async_read_barrier());
    // Nominal quantum even when a_live < a_tile_rows: keeps the CB ring cycle closed.
    cb_push_back(cb_input_rows, a_tile_rows);
}

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t core_start_unit = get_arg_val<uint32_t>(0);
    const uint32_t core_num_units = get_arg_val<uint32_t>(1);

    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);
    Cursor cursor(core_start_unit, core_num_units);

    for (uint32_t block_idx = 0; block_idx < ZONED_BLOCKS && !cursor.done(); ++block_idx) {
        const uint32_t run = cursor.extent();
        load_rows_block<true>(input, cursor, run);
        cursor.advance(run);
    }
    if (!cursor.done()) {
        DeviceZoneScopedN("PERM-RD-TAIL");
        while (!cursor.done()) {
            const uint32_t run = cursor.extent();
            load_rows_block<false>(input, cursor, run);
            cursor.advance(run);
        }
    }
}
