// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Layouts shared by the host and the kernels of qkv_causal_conv1d_silu.
#pragma once

#include <cstdint>

namespace qkv_conv {

// Kimi-K3 supplies four learned causal-convolution taps; output row r reads input rows r - 3 .. r.
constexpr uint32_t tap_count = 4;
constexpr uint32_t history_rows = tap_count - 1;

// A 32x32 BFLOAT16 tile is four 16x16 faces, [top left, top right, bottom left, bottom right]. A tile row is one face
// row in each of its left and right faces. The tile geometry is spelled out here rather than taken from tt::constants
// so this header stays compiler-neutral (it is compiled by both the host and the kernel toolchains); the host program
// factory static_asserts that the values agree.
constexpr uint32_t tile_height = 32;
constexpr uint32_t tile_width = 32;
constexpr uint32_t face_height = 16;
constexpr uint32_t face_width = 16;
constexpr uint32_t element_bytes = sizeof(uint16_t);
constexpr uint32_t face_rows = face_height;
constexpr uint32_t face_row_bytes = face_width * element_bytes;
constexpr uint32_t face_bytes = face_height * face_width * element_bytes;
constexpr uint32_t tile_bytes = tile_height * tile_width * element_bytes;
constexpr uint32_t tile_row_bytes = tile_width * element_bytes;

// Alignment of every DRAM and L1 read the kernels issue; the smallest read holding a face row is the aligned run of
// face rows around it.
constexpr uint32_t read_alignment = 64;
constexpr uint32_t face_rows_per_read = read_alignment / face_row_bytes;

constexpr uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Tiles per destination acquire: the largest divisor of `tiles` within half of the destination registers, which hold
// half as many FP32 tiles as BFLOAT16 ones.
constexpr uint32_t max_dest_tiles = 8;
constexpr uint32_t max_dest_tiles_fp32 = max_dest_tiles / 2;
constexpr uint32_t dest_tile_limit(bool fp32_dest_acc_en) {
    return fp32_dest_acc_en ? max_dest_tiles_fp32 : max_dest_tiles;
}
constexpr uint32_t dest_group(uint32_t tiles, uint32_t limit = max_dest_tiles) {
    uint32_t group = limit;
    while (tiles % group != 0) {
        --group;
    }
    return group;
}

// The reader stages each channel tile as a left and a right strip of face rows: the three rows before the tile, then
// its 32 rows. A strip starts strip_phase bytes past an aligned address, so its row history_rows, where the tile's
// faces land, is aligned.
constexpr uint32_t strip_rows = history_rows + tile_height;
constexpr uint32_t strip_stride = align_up(strip_rows * face_row_bytes, read_alignment);
constexpr uint32_t strip_phase = read_alignment - (history_rows * face_row_bytes) % read_alignment;

// Reader scratch, from an aligned base: two parities of [left, right] strips per channel tile, the previous tile's two
// bottom faces per channel tile, and the three carried rows of the channel block.
struct ReaderScratch {
    uint32_t block_ct;
    constexpr uint32_t staging() const { return 2 * block_ct * 2 * strip_stride + read_alignment; }
    constexpr uint32_t rows() const { return staging() + block_ct * 2 * face_bytes; }
    // Slack for aligning the base.
    constexpr uint32_t bytes() const { return rows() + history_rows * block_ct * tile_row_bytes + read_alignment; }
};

// Work item `work` is tile row work % tile_rows of channel block work / tile_rows. With defer_first_rows each core
// handles a block's tile row 0, which may wait for the predecessor's rows, after the block's other rows, so no block
// is visited twice. On a rank holding a split interval, the tail's first tile row `split_row` (nonzero) waits for the
// predecessor's rows instead of row 0 (which starts from the caller's history): it is deferred the same way, ahead of a
// deferred row 0. Compute uses only a work item's block, so it may walk the order without the split row; the reader and
// writer must walk the same order.
template <typename Process>
inline void for_each_work_item(
    uint32_t first,
    uint32_t count,
    uint32_t tile_rows,
    bool defer_first_rows,
    uint32_t split_row,
    const Process& process) {
    const uint32_t end = first + count;
    for (uint32_t segment_start = first; segment_start < end;) {
        const uint32_t block_start = segment_start / tile_rows * tile_rows;
        const uint32_t block_end = block_start + tile_rows;
        const uint32_t segment_end = block_end < end ? block_end : end;
        const bool defer_first = defer_first_rows && segment_start == block_start;
        const uint32_t split = block_start + split_row;
        const bool defer_split = split_row != 0 && split >= segment_start && split < segment_end;
        // The segment's other rows in order, then the deferred split row, then the deferred row 0. One call site keeps
        // `process` inlined.
        const uint32_t deferred_begin = segment_end - defer_split - defer_first;
        for (uint32_t index = segment_start; index < segment_end; ++index) {
            uint32_t work = segment_start;
            if (index < deferred_begin) {
                work = index + defer_first;
                work += defer_split && work >= split;
            } else if (defer_split && index == deferred_begin) {
                work = split;
            }
            process(work);
        }
        segment_start = segment_end;
    }
}

template <typename Process>
inline void for_each_work_item(
    uint32_t first, uint32_t count, uint32_t tile_rows, bool defer_first_rows, const Process& process) {
    for_each_work_item(first, count, tile_rows, defer_first_rows, 0, process);
}

// Fabric exchange buffer pages. The predecessor writes rows [predecessor_page, +3) and the final segment's owner
// writes rows [carry_page, +3); the convolution cores stage this rank's outgoing and end rows for the exchange core.
constexpr uint32_t predecessor_page = 0;
constexpr uint32_t carry_page = predecessor_page + history_rows;
constexpr uint32_t outgoing_page = carry_page + history_rows;
constexpr uint32_t end_page = outgoing_page + history_rows;
constexpr uint32_t exchange_rows = end_page + history_rows;
// Compile-time index of a fabric connection the exchange core does not have.
constexpr uint32_t no_fabric_connection = ~0U;
// The exchange core's fabric connections: the SP neighbors, and on a turning 2D axis the links peer routes leave on.
constexpr uint32_t max_fabric_connections = 4;

// Writer scratch for staging the outgoing and end rows of up to `max_tiles` channel tiles: an aligned slot for the
// chronology scalars, then per row and tile the aligned reads holding its left and right face rows, then the packed row
// segments.
struct StagingScratch {
    uint32_t max_tiles;
    static constexpr uint32_t rows = 2 * history_rows;
    constexpr uint32_t reads() const { return read_alignment; }
    constexpr uint32_t read(uint32_t row, uint32_t tile, uint32_t side) const {
        return reads() + ((row * max_tiles + tile) * 2 + side) * read_alignment;
    }
    constexpr uint32_t packed(uint32_t row) const {
        return reads() + rows * max_tiles * 2 * read_alignment + row * max_tiles * tile_row_bytes;
    }
    // Slack for aligning the base.
    constexpr uint32_t bytes() const { return packed(rows) + read_alignment; }
};

}  // namespace qkv_conv
