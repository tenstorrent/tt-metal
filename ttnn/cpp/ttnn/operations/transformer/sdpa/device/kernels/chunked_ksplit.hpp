// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef SDPA_PAIR_SPLIT
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_pair_split.hpp"
#endif

// Chunked (paged) causal prefill with grouped query heads and a K split. Shared by the reader, writer and compute
// kernels, which must agree on every piece's K range or the CB counts desync.
//
// A unit is q_heads_per_unit query heads (sharing one K/V head) times one Q chunk. Its causal keys are cut into
// k_splits slices, and the slices are the work items the cores are given, numbered unit * k_splits + slice. A core
// runs the slices of one unit that it was given back to back as one piece (one contiguous K range), so a unit ends
// up in a few pieces, one per core it spans; each piece writes its normalized output and row statistics to its own
// output plane (its slot among the unit's pieces) and the planes are merged after the program.

struct ChunkedKSplitRange {
    uint32_t lo;  // first K chunk
    uint32_t hi;  // one past the last; lo == hi when empty
};

// The unit's causal K chunks are [0, k_end) with k_end = ceil((q_start_tile + Sq_chunk_t) / Sk_chunk_t) clamped to
// k_num_chunks (the bound the unsplit kernels use). Slice s of S takes [k_end * s / S, k_end * (s + 1) / S): slices
// differ by at most one chunk, the last one holds the diagonal, and a slice is empty only when k_end < S.
// Pair split (SDPA_PAIR_SPLIT): K is the chip's half of the cache, so k_end counts the local tiles visible to the
// unit's last row (sdpa_pair_split::local_k_end, at least one); every kernel sets sdpa_pair_split::g_parity first.
inline uint32_t chunked_ksplit_bound(uint32_t k_end, uint32_t S, uint32_t s) { return k_end * s / S; }

inline uint32_t chunked_unit_k_end(
    uint32_t q_start_tile, uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t k_num_chunks) {
#ifdef SDPA_PAIR_SPLIT
    const uint32_t k_end_tiles = sdpa_pair_split::local_k_end(q_start_tile + Sq_chunk_t, sdpa_pair_split::g_parity);
#else
    const uint32_t k_end_tiles = q_start_tile + Sq_chunk_t;
#endif
    const uint32_t k_end = (k_end_tiles + Sk_chunk_t - 1) / Sk_chunk_t;
    return k_end < k_num_chunks ? k_end : k_num_chunks;
}

// One core's run of a unit's slices: items [item, end) of the core's range that belong to item's unit.
struct ChunkedPiece {
    uint32_t unit;
    uint32_t first_slice;  // slices [first_slice, last_slice]
    uint32_t last_slice;
    uint32_t items;  // last_slice - first_slice + 1
};

inline ChunkedPiece chunked_piece_at(uint32_t item, uint32_t range_end, uint32_t S) {
    const uint32_t unit = item / S;
    const uint32_t unit_end = (unit + 1) * S;
    const uint32_t piece_end = range_end < unit_end ? range_end : unit_end;
    return {unit, item - unit * S, piece_end - 1 - unit * S, piece_end - item};
}

// K chunks of the piece of a unit whose Q chunk starts at q_start_tile.
inline ChunkedKSplitRange chunked_piece_range(
    const ChunkedPiece& piece,
    uint32_t q_start_tile,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t k_num_chunks,
    uint32_t S) {
    const uint32_t k_end = chunked_unit_k_end(q_start_tile, Sq_chunk_t, Sk_chunk_t, k_num_chunks);
    return {chunked_ksplit_bound(k_end, S, piece.first_slice), chunked_ksplit_bound(k_end, S, piece.last_slice + 1)};
}
