// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute reader (rows_tile_scatter): load_block.
// Each input tile of the block lands in its own slot of cb_input_tiles as one read of its live
// prefix, issued from the page base (Blackhole DRAM reads need local and remote address equal
// mod 64, so a mid-tile window is never read). One read barrier and one push per block.
// Raw dataflow: no kernel_lib helper reads a byte prefix of a TensorAccessor page into a CB.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"

using namespace g6_permute;

constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
constexpr uint32_t tensor_a = get_compile_time_arg_val(1);
constexpr uint32_t tensor_b = get_compile_time_arg_val(2);
constexpr uint32_t tensor_bt = get_compile_time_arg_val(4);
constexpr uint32_t tensor_wt = get_compile_time_arg_val(5);
constexpr uint32_t block_in_tiles = get_compile_time_arg_val(6);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(7);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(8);
// Args 3 (tensor_at) and 9 (stagger_rows) belong to the writer; the layout is shared.
constexpr auto input_args = TensorAccessorArgs<10>();

using Geometry = TileGeometry<face_row_bytes>;
using Rows = RowsLive<tensor_b, tensor_bt>;
using Cursor = InTileCursor<tensor_a, tensor_bt, tensor_wt>;
static_assert(Geometry::tile_bytes == in_page_bytes, "tile page must be four 16x16 faces");

constexpr uint32_t full_prefix_bytes = Geometry::prefix_bytes(TILE_ROWS);
constexpr uint32_t last_prefix_bytes = Geometry::prefix_bytes(Rows::last);

template <typename Accessor>
FORCE_INLINE void issue_block_reads(const Accessor& input, uint32_t& page, Cursor& cursor, uint32_t extent) {
    uint32_t slot = get_write_ptr(cb_input_tiles);
    for (uint32_t k = 0; k < extent; ++k) {
        const uint32_t bytes = cursor.bt == tensor_bt - 1 ? last_prefix_bytes : full_prefix_bytes;
        noc_async_read(input.get_noc_addr(page), slot, bytes);
        slot += in_page_bytes;
        ++page;
        cursor.advance();
    }
}

template <bool zoned, typename Accessor>
FORCE_INLINE void load_block(const Accessor& input, uint32_t& page, Cursor& cursor, uint32_t extent) {
    G6_ZONED_STEP(zoned, "PERM-RD-RESERVE", cb_reserve_back(cb_input_tiles, block_in_tiles));
    G6_ZONED_STEP(zoned, "PERM-RD-ISSUE", issue_block_reads(input, page, cursor, extent));
    G6_ZONED_STEP(zoned, "PERM-RD-BARRIER", noc_async_read_barrier());
    // Nominal count on a ragged last block too: the writer waits and pops the same quantum.
    cb_push_back(cb_input_tiles, block_in_tiles);
}

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    uint32_t page = get_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    Cursor cursor{
        get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5)};

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);

    uint32_t block_idx = 0;
    for (; block_idx < ZONED_BLOCKS && remaining > 0; ++block_idx) {
        DeviceZoneScopedN("PERM-RD-LOAD-BLOCK");
        const uint32_t extent = remaining < block_in_tiles ? remaining : block_in_tiles;
        load_block<true>(input, page, cursor, extent);
        remaining -= extent;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-RD-TAIL");
        while (remaining > 0) {
            const uint32_t extent = remaining < block_in_tiles ? remaining : block_in_tiles;
            load_block<false>(input, page, cursor, extent);
            remaining -= extent;
        }
    }
}
