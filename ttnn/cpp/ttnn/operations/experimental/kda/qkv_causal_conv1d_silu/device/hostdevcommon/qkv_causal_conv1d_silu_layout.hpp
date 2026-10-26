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

// The shifted tap tiles are four face copies each out of the strips, and the copies cost per NoC transaction rather
// than per byte. The reader makes taps [0, reader_taps) on its NoC and the writer the rest on the other NoC, so the
// two NoCs share them. Compute takes the reader's taps from act_tile and the writer's from act_tile_writer.
constexpr uint32_t reader_taps = tap_count / 2;
constexpr uint32_t writer_taps = tap_count - reader_taps;

// One work item's strips, [left, right] per channel tile, in an entry of the strips dataflow buffer: the reader
// produces it, its own and the writer's tap copies read it, and the writer frees it. The entry starts at any
// L1-aligned address, so it carries the slack for aligning its base.
struct StripEntry {
    uint32_t block_ct;
    constexpr uint32_t bytes() const { return 2 * block_ct * strip_stride + read_alignment; }
    static constexpr uint32_t base(uint32_t entry_address) { return align_up(entry_address, read_alignment); }
    static constexpr uint32_t strip(uint32_t base, uint32_t ct, uint32_t side) {
        return base + strip_phase + (ct * 2 + side) * strip_stride;
    }
};

// Face f of tap t's shifted tile is strip f % 2 from row t + 16 * (f / 2).
constexpr uint32_t shifted_face_source(uint32_t base, uint32_t ct, uint32_t tap, uint32_t face) {
    return StripEntry::strip(base, ct, face % 2) + (tap + (face / 2) * face_rows) * face_row_bytes;
}

// Reader scratch, from an aligned base: the previous tile's two bottom faces per channel tile, then the three carried
// rows of the channel block.
struct ReaderScratch {
    uint32_t block_ct;
    constexpr uint32_t rows() const { return block_ct * 2 * face_bytes; }
    // Slack for aligning the base.
    constexpr uint32_t bytes() const { return rows() + history_rows * block_ct * tile_row_bytes + read_alignment; }
};

// Work item `work` is tile row work % tile_rows of channel block work / tile_rows. Each core walks its items in order,
// except that its items on tile row deferred_row run last, in reverse order: each starts a segment that waits for the
// predecessor's rows, so the core computes everything else first, even when its range crosses a block boundary after
// such an item. Reversed, the deferred item of the block the core ends on comes first and needs no block change; a
// deferred item of an earlier block revisits that block. The reader, writer and compute must all walk this order.
constexpr uint32_t no_deferred_row = ~0U;
template <typename Process>
inline void for_each_work_item(
    uint32_t first, uint32_t count, uint32_t tile_rows, uint32_t deferred_row, const Process& process) {
    const uint32_t end = first + count;
    const auto deferred = [&](uint32_t index) { return index % tile_rows == deferred_row; };
    for (uint32_t index = first; index < end; ++index) {
        if (!deferred(index)) {
            process(index);
        }
    }
    if (deferred_row < tile_rows) {
        for (uint32_t index = end; index > first; --index) {
            if (deferred(index - 1)) {
                process(index - 1);
            }
        }
    }
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
