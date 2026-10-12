// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Layouts shared by the host and the kernels of qkv_causal_conv1d_silu.
#pragma once

#include <cstdint>

namespace qkv_conv {

constexpr uint32_t history_rows = 3;

// The reader stages each channel tile as left and right strips of 3 + 32 face rows, one strip_stride apart.
constexpr uint32_t strip_stride = 1152;

// Work item `work` is tile row work % tile_rows of channel block work / tile_rows. With defer_first_rows each core
// handles a block's tile row 0, which may wait for the predecessor's rows, after the block's other rows, so no block
// is visited twice; the reader, compute and writer must all walk this order.
template <typename Process>
inline void for_each_work_item(
    uint32_t first, uint32_t count, uint32_t tile_rows, bool defer_first_rows, Process&& process) {
    const uint32_t end = first + count;
    uint32_t segment_start = first;
    for (uint32_t index = first; index < end; ++index) {
        const uint32_t block_end = (segment_start / tile_rows + 1) * tile_rows;
        const uint32_t segment_end = block_end < end ? block_end : end;
        const bool defer = defer_first_rows && segment_start % tile_rows == 0;
        // A deferred segment runs rows [1, n) first, then its row 0.
        process(defer ? (index + 1 == segment_end ? segment_start : index + 1) : index);
        if (index + 1 == segment_end) {
            segment_start = segment_end;
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

// A tile row is one 32-byte face row in each of its left and right faces; the smallest DRAM read holding a face row
// is the 64-byte aligned pair of face rows around it.
constexpr uint32_t face_row_bytes = 32;
constexpr uint32_t face_bytes = 16 * face_row_bytes;
constexpr uint32_t face_row_pair_bytes = 2 * face_row_bytes;

// Reader scratch for staging `rows` exchanged rows of up to `max_tiles` channel tiles: per row and tile, the left and
// right face-row pairs, then the packed row segment.
constexpr uint32_t staged_rows_scratch_bytes(uint32_t rows, uint32_t max_tiles) {
    return rows * max_tiles * 3 * face_row_pair_bytes;
}

}  // namespace qkv_conv
