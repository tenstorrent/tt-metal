// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two-part readiness of a device's own slice, shared by the ring-attention all-gather (which sends the slice in this
// order with a marker between the parts) and its fused ring-joint SDPA consumer (which starts on part 0 at the first
// remote ring step). The slice's K chunks are split into bands like the consumer's K split (band i owns
// [i * n / bands, (i + 1) * n / bands)), and part 0 is the leading share of every band, so every band has work as soon
// as part 0 lands. Part 1 is the rest of every band, then the tail of rows past the split chunks.
//
// Both sides compute the split from the same values (the gather's slice extent, the truncation for this receiver, the
// chunk size and band count) with these functions, and the consumer decides which chunks part 0 covers from the
// sender's split, never its own K split: a disagreement between the two can only cost overlap, not read data early.

#pragma once

#include <cstdint>

namespace ttnn::ring_attention_all_gather {

// Part 0 is this share of every band (rounded): it must land before the receiver finishes its local ring step, and part
// 1 before the receiver has worked through part 0. 5/8 balances the two for the Kimi shapes on an 8-device ring.
constexpr uint32_t kLocalReadinessPart0Eighths = 5;

struct ChunkRange {
    uint32_t begin;
    uint32_t end;
};

// K chunks the sender splits into bands for one receiver. A receiver ranked below the sender masks the sender's newest
// slab entirely (dense causal skip, truncate_tiles > 0), so the split leaves it out; those rows still travel in the
// tail of part 1, since they are relayed on to devices that need them.
inline uint32_t local_readiness_split_chunks(
    uint32_t valid_rows, uint32_t truncate_tiles, bool receiver_masks_newest_slab, uint32_t chunk_tiles) {
    const uint32_t dropped = receiver_masks_newest_slab ? truncate_tiles : 0;
    const uint32_t split_rows = valid_rows > dropped ? valid_rows - dropped : 0;
    return (split_rows + chunk_tiles - 1) / chunk_tiles;
}

// Chunks of `part` (0 or 1) in `band` of a split of `num_chunks` K chunks into `bands`.
inline ChunkRange local_readiness_chunks(uint32_t num_chunks, uint32_t bands, uint32_t band, uint32_t part) {
    const uint32_t begin = band * num_chunks / bands;
    const uint32_t end = (band + 1) * num_chunks / bands;
    const uint32_t mid = begin + ((end - begin) * kLocalReadinessPart0Eighths + 4) / 8;
    return part == 0 ? ChunkRange{begin, mid} : ChunkRange{mid, end};
}

// Whether `chunk` arrives in part 0 of a split of `num_chunks` chunks into `bands`.
inline bool local_readiness_in_part0(uint32_t chunk, uint32_t num_chunks, uint32_t bands) {
    for (uint32_t band = 0; band < bands; ++band) {
        const ChunkRange r = local_readiness_chunks(num_chunks, bands, band, 0);
        if (chunk >= r.begin && chunk < r.end) {
            return true;
        }
    }
    return false;
}

// Row ranges of `part` in sending order, for one input of `valid_rows` rows: range r < bands is band r's share, range
// r == bands is the tail past the split chunks (part 1 only; empty in part 0). Over both parts every row in
// [0, valid_rows) is covered exactly once. The gather reader and writer both walk the slice as
// `for (r = 0; r <= bands; ++r)` over these, which keeps their page order identical. (A plain range function rather
// than a visitor: a lambda around the per-packet send loop slowed the gather.)
struct RowRange {
    uint32_t begin;
    uint32_t end;
};
inline RowRange local_readiness_rows(
    uint32_t num_chunks, uint32_t bands, uint32_t chunk_tiles, uint32_t valid_rows, uint32_t part, uint32_t range) {
    const auto clamp = [valid_rows](uint32_t rows) { return rows < valid_rows ? rows : valid_rows; };
    if (range == bands) {
        return part == 0 ? RowRange{0, 0} : RowRange{clamp(num_chunks * chunk_tiles), valid_rows};
    }
    const ChunkRange chunks = local_readiness_chunks(num_chunks, bands, range, part);
    return {clamp(chunks.begin * chunk_tiles), clamp(chunks.end * chunk_tiles)};
}

}  // namespace ttnn::ring_attention_all_gather
