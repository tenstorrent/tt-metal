// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Rotating KV window for a fused ring-joint SDPA consumer: instead of a buffer holding every gathered slice, each
// direction lands its slices' readiness parts (ring_attention_slice_readiness.hpp) in a ring of W part-sized slots,
// and the diametric slice's parts, half from each direction, in a ring of Wd shared slots. Unit n = k * kReadinessParts
// + p of a direction (part p of the k-th slice received in it) lands in slot dir * W + n % W; diametric part p in slot
// 2W + p % Wd. A slot is reused once every consumer has passed the unit in it: the SDPA cores and the gather reader
// that relays it publish how far they got, and the opposite direction's writer returns credits upstream.
//
// Within a slot, part p of a slice is packed in send order: range r (a band's share) starts at the rows of the part's
// earlier ranges. Every receiver splits a slice the same way, so a relay moves part p to part p unchanged.
//
// Liveness needs W > kReadinessParts: a device frees its k-th slice's parts only once the downstream has stored them as
// its (k+1)-th, and the consumer uses all of slice k before slice k + 1. The relayed diametric half sits in a rotating
// slot until the downstream consumes it, last, so W + Wd >= kReadinessParts as well.

#pragma once

#include <cstdint>

#include "ring_attention_slice_readiness.hpp"

namespace ttnn::ring_attention_all_gather {

// Fields of a consumer's published progress word: the units it is done with in each direction, and the diametric parts.
constexpr uint32_t kKvWindowFieldBits = 10;
constexpr uint32_t kKvWindowFieldMask = (1u << kKvWindowFieldBits) - 1;
constexpr uint32_t kKvWindowDiamShift = 2 * kKvWindowFieldBits;
// A field no consumer of this word tracks.
constexpr uint32_t kKvWindowDone = kKvWindowFieldMask;

inline uint32_t kv_window_progress(uint32_t dir0, uint32_t dir1, uint32_t diam) {
    return dir0 | dir1 << kKvWindowFieldBits | diam << kKvWindowDiamShift;
}
inline uint32_t kv_window_field(uint32_t progress, uint32_t field) {
    return (progress >> (field * kKvWindowFieldBits)) & kKvWindowFieldMask;
}

// Credits a writer receives, one word: rotating units in the low half, diametric parts in the high half.
constexpr uint32_t kKvWindowDiamCreditShift = 16;

inline uint32_t kv_window_rot_slot(uint32_t dir, uint32_t unit, uint32_t window_parts) {
    return dir * window_parts + unit % window_parts;
}
inline uint32_t kv_window_diam_slot(uint32_t part, uint32_t window_parts, uint32_t diam_parts) {
    return 2 * window_parts + part % diam_parts;
}

// Tile rows a slot needs for slices of at most max_rows rows: every band's share of a part, at most.
inline uint32_t kv_window_slot_rows(uint32_t max_rows, uint32_t bands, uint32_t chunk_tiles) {
    const uint32_t chunks = (max_rows + chunk_tiles - 1) / chunk_tiles;
    const uint32_t band_chunks = (chunks + bands - 1) / bands;
    return bands * ((band_chunks + kReadinessParts - 1) / kReadinessParts) * chunk_tiles;
}

// First slot row of range `range` of `part`: the rows of the part's earlier ranges.
inline uint32_t kv_window_range_row(
    uint32_t num_chunks,
    uint32_t bands,
    uint32_t chunk_tiles,
    uint32_t valid_rows,
    uint32_t part,
    uint32_t range,
    uint32_t key) {
    uint32_t row = 0;
    for (uint32_t r = 0; r < range; ++r) {
        const RowRange rows = readiness_rows(num_chunks, bands, chunk_tiles, valid_rows, part, r, key);
        row += rows.end > rows.begin ? rows.end - rows.begin : 0;
    }
    return row;
}

// Where K chunk `chunk` of a slice sits: its part, its first slot row and the end of its range's rows (rows past it
// belong to the next range, so a reader zero-fills them).
struct KvWindowChunk {
    uint32_t part;
    uint32_t row;
    uint32_t row_end;
};

inline KvWindowChunk kv_window_locate(
    uint32_t chunk, uint32_t num_chunks, uint32_t bands, uint32_t chunk_tiles, uint32_t valid_rows, uint32_t key) {
    uint32_t band = 0;
    while (band + 1 < bands && chunk >= band_begin(num_chunks, bands, band + 1, key)) {
        ++band;
    }
    uint32_t part = 0;
    while (part + 1 < kReadinessParts && chunk >= readiness_chunks(num_chunks, bands, band, part, key).end) {
        ++part;
    }
    const uint32_t row = kv_window_range_row(num_chunks, bands, chunk_tiles, valid_rows, part, band, key);
    const RowRange rows = readiness_rows(num_chunks, bands, chunk_tiles, valid_rows, part, band, key);
    return {part, row + chunk * chunk_tiles - rows.begin, row + rows.end - rows.begin};
}

}  // namespace ttnn::ring_attention_all_gather
