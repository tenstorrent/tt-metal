// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute writer (rows_tile_scatter): scatter_rows_block.
// Tile row r of input tile (n, a, bt, wt) is output tile row a mod 32 of output tile
// (n, 32*bt + r, a div 32, wt); its two 32-B halves go straight from the input slot to that
// output page in DRAM. No compute engine touches the values, so every bf16 bit survives.
// Raw dataflow: kernel_lib has no block helper that scatters tile rows to remote pages.

#include "api/dataflow/dataflow_api.h"
#include "g6_permute_common.hpp"

using namespace g6_permute;

constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
constexpr uint32_t tensor_a = get_compile_time_arg_val(1);
constexpr uint32_t tensor_b = get_compile_time_arg_val(2);
constexpr uint32_t tensor_at = get_compile_time_arg_val(3);
constexpr uint32_t tensor_bt = get_compile_time_arg_val(4);
constexpr uint32_t tensor_wt = get_compile_time_arg_val(5);
constexpr uint32_t block_in_tiles = get_compile_time_arg_val(6);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(7);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(8);
constexpr bool stagger_rows = get_compile_time_arg_val(9) != 0;
constexpr auto output_args = TensorAccessorArgs<10>();

using Geometry = TileGeometry<face_row_bytes>;
using Rows = RowsLive<tensor_b, tensor_bt>;
using Cursor = InTileCursor<tensor_a, tensor_bt, tensor_wt>;
static_assert(Geometry::tile_bytes == in_page_bytes, "tile page must be four 16x16 faces");

// Consecutive rows r of one input tile land in output pages this far apart, all at the same row.
constexpr uint32_t out_row_page_stride = tensor_at * tensor_wt;

template <typename Accessor>
FORCE_INLINE void scatter_tile(const Accessor& output, uint32_t slot, const Cursor& c) {
    const bool last_bt = c.bt == tensor_bt - 1;
    const uint32_t rows = last_bt ? Rows::last : TILE_ROWS;
    const uint32_t dst_offset = Geometry::face_off(c.a % TILE_ROWS, 0);
    const uint32_t base_page = ((c.n * tensor_b + TILE_ROWS * c.bt) * tensor_at + c.a / TILE_ROWS) * tensor_wt + c.wt;
    // Cores holding neighbouring a start on different rows, so their first writes hit different pages.
    uint32_t r = 0;
    if constexpr (stagger_rows) {
        r = last_bt ? c.a % Rows::last : c.a % TILE_ROWS;
    }
    for (uint32_t i = 0; i < rows; ++i) {
        const uint64_t dst = output.get_noc_addr(base_page + r * out_row_page_stride, dst_offset);
        const uint32_t src = slot + Geometry::face_off(r, 0);
        noc_async_write_one_packet_set_state(dst, face_row_bytes);
        noc_async_write_one_packet_with_state(src, static_cast<uint32_t>(dst));
        noc_async_write_one_packet_with_state(
            src + Geometry::face_bytes, static_cast<uint32_t>(dst) + Geometry::face_bytes);
        if (++r == rows) {
            r = 0;
        }
    }
}

template <typename Accessor>
FORCE_INLINE void issue_block_writes(const Accessor& output, Cursor& cursor, uint32_t extent) {
    uint32_t slot = get_read_ptr(cb_input_tiles);
    for (uint32_t k = 0; k < extent; ++k) {
        scatter_tile(output, slot, cursor);
        slot += in_page_bytes;
        cursor.advance();
    }
}

template <bool zoned, typename Accessor>
FORCE_INLINE void scatter_rows_block(const Accessor& output, Cursor& cursor, uint32_t extent) {
    G6_ZONED_STEP(zoned, "PERM-WR-WAIT", cb_wait_front(cb_input_tiles, block_in_tiles));
    G6_ZONED_STEP(zoned, "PERM-WR-ISSUE", issue_block_writes(output, cursor, extent));
    // The slots are the writes' source: they must have left L1 before the reader may refill them.
    G6_ZONED_STEP(zoned, "PERM-WR-FLUSH", noc_async_writes_flushed());
    cb_pop_front(cb_input_tiles, block_in_tiles);
}

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    uint32_t remaining = get_arg_val<uint32_t>(1);
    Cursor cursor{
        get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5)};

    const auto output = TensorAccessor(output_args, output_addr, in_page_bytes);

    uint32_t block_idx = 0;
    for (; block_idx < ZONED_BLOCKS && remaining > 0; ++block_idx) {
        DeviceZoneScopedN("PERM-WR-SCATTER-BLOCK");
        const uint32_t extent = remaining < block_in_tiles ? remaining : block_in_tiles;
        scatter_rows_block<true>(output, cursor, extent);
        remaining -= extent;
    }
    if (remaining > 0) {
        DeviceZoneScopedN("PERM-WR-TAIL");
        while (remaining > 0) {
            const uint32_t extent = remaining < block_in_tiles ? remaining : block_in_tiles;
            scatter_rows_block<false>(output, cursor, extent);
            remaining -= extent;
        }
    }

    // The next program starts right after this one: no write may still be in flight.
    {
        DeviceZoneScopedN("PERM-WR-FINAL-BARRIER");
        noc_async_write_barrier();
    }
}
