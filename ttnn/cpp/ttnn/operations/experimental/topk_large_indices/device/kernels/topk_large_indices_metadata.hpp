// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "topk_large_indices_compute_body_mode.hpp"

struct TopkMetadataBounds {
    uint32_t num_chunks;
    uint32_t tail_elements;
};

constexpr TopkMetadataBounds calculate_topk_bounds(uint32_t search_len, uint32_t chunk_elements) {
    const uint32_t num_chunks = (search_len + chunk_elements - 1) / chunk_elements;
    return {
        .num_chunks = num_chunks,
        .tail_elements = search_len - ((num_chunks - 1) * chunk_elements),
    };
}

// A segment past the row's valid length is reduced as one all -inf chunk, so the body never waits on an empty stream.
struct TopkSegmentBounds {
    uint32_t first_chunk;
    uint32_t num_chunks;
    uint32_t tail_elements;
    bool empty;
};

constexpr TopkSegmentBounds calculate_topk_segment_bounds(
    TopkMetadataBounds row, uint32_t chunk_elements, uint32_t seg_first_chunk, uint32_t seg_end_chunk) {
    const uint32_t end = seg_end_chunk < row.num_chunks ? seg_end_chunk : row.num_chunks;
    const uint32_t first = seg_first_chunk < end ? seg_first_chunk : end;
    if (first == end) {
        return {.first_chunk = first, .num_chunks = 1, .tail_elements = chunk_elements, .empty = true};
    }
    return {
        .first_chunk = first,
        .num_chunks = end - first,
        .tail_elements = end == row.num_chunks ? row.tail_elements : chunk_elements,
        .empty = false,
    };
}

// The half a core owns, cut from the row's real chunk count. With a metadata length only the device knows that
// count, so the split point is derived here rather than sent from the host -- otherwise the halves are cut from
// the physical width and the second one is mostly clipped away.
//
// The cut lands on a 32-chunk block boundary because the body reduces a half in 32-chunk segments anchored at
// chunk 0: a half starting mid-block straddles one block more than it needs, and every extra segment costs an
// unfused merge and rebuild. Balancing BLOCKS rather than chunks leaves each core ceil(blocks / 2) segments,
// which is the least either can run. Rows of a single block have nothing to align and split by chunk.
constexpr TopkSegmentBounds half_bounds(TopkMetadataBounds row, uint32_t chunk_elements, bool second_half) {
    namespace program = ttnn::operations::experimental::topk_large_indices::program;
    const uint32_t blocks = (row.num_chunks + program::max_fused_chunks - 1) / program::max_fused_chunks;
    const uint32_t mid = blocks >= 2 ? (blocks / 2) * program::max_fused_chunks : row.num_chunks / 2;
    return calculate_topk_segment_bounds(
        row, chunk_elements, second_half ? mid : 0, second_half ? row.num_chunks : mid);
}

// A core's whole rows are always the whole row, never a chunk range.
constexpr TopkSegmentBounds whole_row_bounds(TopkMetadataBounds row) {
    return {.first_chunk = 0, .num_chunks = row.num_chunks, .tail_elements = row.tail_elements, .empty = false};
}

// The mailbox holds the whole-row bounds, then the split row's half bounds.
inline constexpr uint32_t topk_metadata_split_word_offset = sizeof(TopkMetadataBounds) / sizeof(uint32_t);

inline constexpr uint32_t topk_metadata_num_chunks_word = offsetof(TopkMetadataBounds, num_chunks) / sizeof(uint32_t);
inline constexpr uint32_t topk_metadata_tail_elements_word =
    offsetof(TopkMetadataBounds, tail_elements) / sizeof(uint32_t);

static_assert(sizeof(TopkMetadataBounds) == 2 * sizeof(uint32_t));
