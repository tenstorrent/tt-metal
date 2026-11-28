// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute reader (nhwc_tile_bfp8_transpose): load_group.
// Group (n, ht, wt, ct) needs input tiles (n, h, wt, ct) for h in [32 * ht, 32 * ht + h_live): slot
// h mod 32 of cb_input_tiles. The group is pushed in chunks of read_chunk_tiles slots, every read of
// a chunk before one barrier, so compute transposes the first chunk while the rest is in flight. A
// short last tile row (H not a multiple of 32) leaves the trailing slots stale; they only reach
// output padding rows, each a bfloat8_b group of its own.
// Raw dataflow: no kernel_lib helper reads TensorAccessor pages into a CB.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "g6_permute_nhwc_bfp8_common.hpp"

using namespace g6_permute_nhwc;

constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
constexpr uint32_t tensor_h = get_compile_time_arg_val(1);
constexpr uint32_t tensor_w = get_compile_time_arg_val(2);
constexpr uint32_t tensor_ht = get_compile_time_arg_val(4);
constexpr uint32_t tensor_wt = get_compile_time_arg_val(5);
constexpr uint32_t tensor_ct = get_compile_time_arg_val(6);
constexpr uint32_t in_page_bytes = get_compile_time_arg_val(7);
constexpr uint32_t read_chunk_tiles = get_compile_time_arg_val(8);
// Arg 3 (tensor_c) belongs to the writer; the layout is shared.
constexpr auto input_args = TensorAccessorArgs<9>();

static_assert(in_page_bytes == BFP8_TILE_BYTES, "input pages are 32x32 bfloat8_b tiles");
static_assert(read_chunk_tiles >= 1 && GROUP_TILES % read_chunk_tiles == 0, "a group is whole chunks");

using Cursor = GroupCursor<tensor_ht, tensor_wt, tensor_ct>;

// Consecutive h of one (n, wt, ct) are this many input pages apart.
constexpr uint32_t h_page_stride = tensor_wt * tensor_ct;

// Input rows w >= tensor_w of the last tile column turn into output padding columns that share
// 16-value groups with live columns. A zero mantissa unpacks to 0 whatever the row's exponent, so
// zeroing their mantissas keeps a padding value from raising a group's shared exponent.
constexpr uint32_t w_tail = tensor_w % GROUP_TILES;

// Mantissa bytes of rows [first_row, 16) of each face of one face pair (0: faces 0/1, rows 0-15;
// 1: faces 2/3, rows 16-31) are one contiguous run per face.
FORCE_INLINE void zero_face_pair_rows(
    const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset, uint32_t pair, uint32_t first_row) {
    const uint32_t bytes = (FACE_ROWS - first_row) * BFP8_FACE_ROW_BYTES;
    for (uint32_t face = 2 * pair; face < 2 * pair + 2; ++face) {
        noc.async_write_zeros(cb, bytes, {.offset_bytes = tile_offset + bfp8_mantissa_offset(face, first_row)});
    }
}

// Slots [0, live) of the reserved chunk: offsets are relative to the CB's write pointer.
FORCE_INLINE void zero_padding_rows(const Noc& noc, const CircularBuffer& cb, uint32_t live) {
    for (uint32_t r = 0, offset = 0; r < live; ++r, offset += in_page_bytes) {
        if constexpr (w_tail < FACE_ROWS) {
            zero_face_pair_rows(noc, cb, offset, 0, w_tail);
            zero_face_pair_rows(noc, cb, offset, 1, 0);
        } else {
            zero_face_pair_rows(noc, cb, offset, 1, w_tail - FACE_ROWS);
        }
    }
    noc.write_zeros_l1_barrier();
}

template <typename Accessor>
FORCE_INLINE void load_group(const Noc& noc, const CircularBuffer& cb, const Accessor& input, const Cursor& g) {
    const uint32_t h_live = live_extent(tensor_h, g.ht);
    uint32_t page = ((g.n * tensor_h + GROUP_TILES * g.ht) * tensor_wt + g.wt) * tensor_ct + g.ct;
    for (uint32_t base = 0; base < GROUP_TILES; base += read_chunk_tiles) {
        // Slots past h_live are pushed unread: compute consumes whole groups.
        const uint32_t live =
            base >= h_live ? 0 : (h_live - base < read_chunk_tiles ? h_live - base : read_chunk_tiles);
        cb_reserve_back(cb_input_tiles, read_chunk_tiles);
        uint32_t slot = get_write_ptr(cb_input_tiles);
        for (uint32_t r = 0; r < live; ++r) {
            noc_async_read(input.get_noc_addr(page), slot, in_page_bytes);
            page += h_page_stride;
            slot += in_page_bytes;
        }
        noc_async_read_barrier();
        if constexpr (w_tail != 0) {
            if (g.wt == tensor_wt - 1) {
                zero_padding_rows(noc, cb, live);
            }
        }
        cb_push_back(cb_input_tiles, read_chunk_tiles);
    }
}

void kernel_main() {
    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(0);
    Cursor cursor{
        get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);
    const Noc noc;
    const CircularBuffer cb(cb_input_tiles);

    for (uint32_t g = 0; g < num_groups; ++g) {
        DeviceZoneScopedN("PERM-NHWC-RD-GROUP");
        load_group(noc, cb, input, cursor);
        cursor.advance();
    }
}
