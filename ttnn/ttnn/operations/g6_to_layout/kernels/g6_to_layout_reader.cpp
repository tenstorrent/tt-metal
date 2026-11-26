// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout reader (RISCV_1 / NoC0) — block operation `load_block`.
//
// Streams this core's contiguous segment range [start_segment, start_segment + num_segments)
// into cb_tiles, block_segments segments (block_segments * segment_width tile slots) per
// reserve / barrier / push. Only the live face rows of each tile are read: rows past H and the
// right faces of a tile whose live width is <= 16 are tile padding and never leave DRAM.
//
// No kernel_lib helper reads tile pages (let alone sub-tile live ranges), so this is raw NoC.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

// At most four face-row ranges per tile; adjacent ones merge, so a fully live tile is one read.
struct LiveRanges {
    uint32_t offset[4];
    uint32_t bytes[4];
    uint32_t count;
};

template <uint32_t face_bytes>
LiveRanges live_ranges(uint32_t rows_live, bool right_faces_live) {
    constexpr uint32_t face_row_bytes = face_bytes / FACE_DIM;
    const uint32_t top = rows_live < FACE_DIM ? rows_live : FACE_DIM;
    const uint32_t bottom = rows_live - top;
    LiveRanges ranges{};
    auto add = [&](uint32_t offset, uint32_t bytes) {
        if (bytes == 0) {
            return;
        }
        if (ranges.count > 0 && ranges.offset[ranges.count - 1] + ranges.bytes[ranges.count - 1] == offset) {
            ranges.bytes[ranges.count - 1] += bytes;
        } else {
            ranges.offset[ranges.count] = offset;
            ranges.bytes[ranges.count] = bytes;
            ++ranges.count;
        }
    };
    add(0, top * face_row_bytes);
    if (right_faces_live) {
        add(face_bytes, top * face_row_bytes);
    }
    add(2 * face_bytes, bottom * face_row_bytes);
    if (right_faces_live) {
        add(3 * face_bytes, bottom * face_row_bytes);
    }
    return ranges;
}

// Each range lands at its own offset of the tile's slot, which keeps local == DRAM address mod 64.
template <typename Accessor>
FORCE_INLINE void read_tile(const Accessor& input, uint32_t tile_id, uint32_t slot_addr, const LiveRanges& ranges) {
    for (uint32_t k = 0; k < ranges.count; ++k) {
        noc_async_read(input.get_noc_addr(tile_id, ranges.offset[k]), slot_addr + ranges.offset[k], ranges.bytes[k]);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_tiles_id = get_compile_time_arg_val(0);
    constexpr uint32_t segment_width = get_compile_time_arg_val(1);
    constexpr uint32_t block_segments = get_compile_time_arg_val(2);
    constexpr uint32_t num_col_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_h_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t height = get_compile_time_arg_val(6);
    constexpr uint32_t width = get_compile_time_arg_val(7);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(8);
    constexpr auto input_args = TensorAccessorArgs<9>();

    constexpr uint32_t face_bytes = tile_bytes / 4;
    constexpr uint32_t last_col_live = width - (tensor_w_tiles - 1) * TILE_DIM;
    constexpr bool last_col_right_live = last_col_live > FACE_DIM;
    constexpr uint32_t segment_bytes = segment_width * tile_bytes;

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_segment = get_arg_val<uint32_t>(0);
    const uint32_t num_segments = get_arg_val<uint32_t>(1);

    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);
    DataflowBuffer cb_tiles(cb_tiles_id);

    const uint32_t end_segment = start_segment + num_segments;
    uint32_t segment = start_segment;
    while (segment < end_segment) {
        DeviceZoneScopedN("load_block");
        const uint32_t remaining = end_segment - segment;
        const uint32_t block_segs = remaining < block_segments ? remaining : block_segments;
        const uint32_t block_pages = block_segs * segment_width;
        {
            DeviceZoneScopedN("reserve_tiles");
            cb_tiles.reserve_back(block_pages);
        }
        uint32_t segment_addr = cb_tiles.get_write_ptr();
        for (uint32_t s = 0; s < block_segs; ++s, ++segment, segment_addr += segment_bytes) {
            const uint32_t tile_row = segment / num_col_chunks;
            const uint32_t chunk = segment - tile_row * num_col_chunks;
            const uint32_t ht = tile_row % tensor_h_tiles;
            const uint32_t rows_left = height - ht * TILE_DIM;
            const uint32_t rows_live = rows_left < TILE_DIM ? rows_left : TILE_DIM;
            const uint32_t first_wt = chunk * segment_width;
            const uint32_t cols_left = tensor_w_tiles - first_wt;
            const uint32_t tiles_live = cols_left < segment_width ? cols_left : segment_width;
            const uint32_t first_tile = tile_row * tensor_w_tiles + first_wt;

            // Slots past tiles_live (ragged last chunk) stay stale; the writer never emits them.
            const bool ends_tensor_row = first_wt + tiles_live == tensor_w_tiles;
            const uint32_t interior_tiles = ends_tensor_row ? tiles_live - 1 : tiles_live;
            const LiveRanges interior = live_ranges<face_bytes>(rows_live, true);
            for (uint32_t t = 0; t < interior_tiles; ++t) {
                read_tile(input, first_tile + t, segment_addr + t * tile_bytes, interior);
            }
            if (ends_tensor_row) {
                read_tile(
                    input,
                    first_tile + interior_tiles,
                    segment_addr + interior_tiles * tile_bytes,
                    live_ranges<face_bytes>(rows_live, last_col_right_live));
            }
        }
        {
            DeviceZoneScopedN("read_barrier");
            noc_async_read_barrier();
        }
        cb_tiles.push_back(block_pages);
    }
}
