// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

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

// The chunks of one row that a single core reduces. A column segment past the row's valid length has no
// data; it is reduced as one synthetic all -inf chunk so the fused body never waits on an empty stream.
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

inline constexpr uint32_t topk_metadata_num_chunks_word = offsetof(TopkMetadataBounds, num_chunks) / sizeof(uint32_t);
inline constexpr uint32_t topk_metadata_tail_elements_word =
    offsetof(TopkMetadataBounds, tail_elements) / sizeof(uint32_t);

static_assert(sizeof(TopkMetadataBounds) == 2 * sizeof(uint32_t));
