// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Progressive readiness of a gathered slice, shared by the ring-attention all-gather (which sends every full slice, its
// own and the ones it relays, in this order with a marker after each part) and its fused ring-joint SDPA consumer
// (which waits for a chunk's part, not the whole slice). The slice's K chunks are split into bands like the consumer's
// K split (band i owns [i * n / bands, (i + 1) * n / bands)), and part p is the p-th of kReadinessParts equal shares of
// every band, so every band has work as soon as part 0 lands and each core's chunks arrive in the order it reads them.
// The last part also carries the tail of rows past the split chunks. The diametric slice of split forwarding is sent
// whole, with one marker.
//
// Both sides compute the split from the same values (the gather's slice extent, the truncation for this receiver, the
// chunk size and band count) with these functions, and the consumer decides which part holds a chunk from the
// sender's split, never its own K split: a disagreement between the two can only cost overlap, not read data early.

#pragma once

#include <cstdint>

namespace ttnn::ring_attention_all_gather {

// Parts per full slice: one marker each. More parts let a ring step start sooner after its slice starts landing, at the
// cost of a marker, a write barrier and shorter DRAM bursts per part.
constexpr uint32_t kReadinessParts = 4;

struct ChunkRange {
    uint32_t begin;
    uint32_t end;
};

// K chunks the sender splits into bands for one receiver. A receiver ranked below the sender masks the sender's newest
// slab entirely (dense causal skip, truncate_tiles > 0), so the split leaves it out; those rows still travel in the
// tail of the last part, since they are relayed on to devices that need them.
inline uint32_t readiness_split_chunks(
    uint32_t valid_rows, uint32_t truncate_tiles, bool receiver_masks_newest_slab, uint32_t chunk_tiles) {
    const uint32_t dropped = receiver_masks_newest_slab ? truncate_tiles : 0;
    const uint32_t split_rows = valid_rows > dropped ? valid_rows - dropped : 0;
    return (split_rows + chunk_tiles - 1) / chunk_tiles;
}

// Chunks of `part` in `band` of a split of `num_chunks` K chunks into `bands`.
inline ChunkRange readiness_chunks(uint32_t num_chunks, uint32_t bands, uint32_t band, uint32_t part) {
    const uint32_t begin = band * num_chunks / bands;
    const uint32_t len = (band + 1) * num_chunks / bands - begin;
    return {begin + len * part / kReadinessParts, begin + len * (part + 1) / kReadinessParts};
}

// The part `chunk` arrives in, for a split of `num_chunks` chunks into `bands` (chunks past the split: the last part).
inline uint32_t readiness_part_of(uint32_t chunk, uint32_t num_chunks, uint32_t bands) {
    if (chunk >= num_chunks) {
        return kReadinessParts - 1;
    }
    uint32_t band = 0;
    while (band + 1 < bands && chunk >= (band + 1) * num_chunks / bands) {
        ++band;
    }
    uint32_t part = 0;
    while (part + 1 < kReadinessParts && chunk >= readiness_chunks(num_chunks, bands, band, part).end) {
        ++part;
    }
    return part;
}

// Row ranges of `part` in sending order, for one input of `valid_rows` rows: range r < bands is band r's share, range
// r == bands is the tail past the split chunks (last part only). Over all parts every row in
// [0, valid_rows) is covered exactly once. The gather reader and writer both walk the slice as
// `for (r = 0; r <= bands; ++r)` over these, which keeps their page order identical. (A plain range function rather
// than a visitor: a lambda around the per-packet send loop slowed the gather.)
struct RowRange {
    uint32_t begin;
    uint32_t end;
};
inline RowRange readiness_rows(
    uint32_t num_chunks, uint32_t bands, uint32_t chunk_tiles, uint32_t valid_rows, uint32_t part, uint32_t range) {
    const auto clamp = [valid_rows](uint32_t rows) { return rows < valid_rows ? rows : valid_rows; };
    if (range == bands) {
        return part + 1 < kReadinessParts ? RowRange{0, 0} : RowRange{clamp(num_chunks * chunk_tiles), valid_rows};
    }
    const ChunkRange chunks = readiness_chunks(num_chunks, bands, range, part);
    return {clamp(chunks.begin * chunk_tiles), clamp(chunks.end * chunk_tiles)};
}

}  // namespace ttnn::ring_attention_all_gather
