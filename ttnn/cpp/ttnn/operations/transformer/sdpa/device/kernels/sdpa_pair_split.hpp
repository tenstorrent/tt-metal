// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pair-split chunked prefill (SDPA_PAIR_SPLIT): two chips share a KV head and each caches half of every
// 128-token global block. Chip c (parity, 0 or 1) holds global positions [128 j + 64 c, 128 j + 64 c + 64)
// in its local positions [64 j, 64 j + 64), so local index l is global position
//     G(l, c) = 128 * (l >> 6) + 64 * c + (l & 63).
// In tiles (32 positions): local tile t holds global tile g(t) = 4 * (t >> 1) + 2 * c + (t & 1), contiguous
// within the tile. Queries stay global (query i of the chunk is at chunk_start + i). Local key l is visible
// to the query at p iff G(l, c) <= p, so each (Q tile, local K tile) pair is fully visible, a causal
// diagonal tile, or fully masked, exactly like ordinary causal attention once each global Q tile is mapped
// to a local "row position" (local_q_tile below). The kernels keep their tile-level causal logic and only
// swap the Q-to-K comparison through these helpers.
//
// Only compiled into a call that passes the parity tensor; the default kernels never include this file.

#pragma once

#include <stdint.h>

namespace sdpa_pair_split {

// This device's parity c, set once per call from the per-device parity tensor (compute reads it from the
// chunk-start CB the reader fills).
inline uint32_t g_parity = 0;

// Number of chip-c local K tiles whose global tile index is below global_tile_end.
inline uint32_t local_tiles_before(uint32_t global_tile_end, uint32_t c) {
    const uint32_t rem = global_tile_end & 3u;
    const uint32_t lo = 2u * c;
    const uint32_t partial = rem > lo ? ((rem - lo) > 2u ? 2u : (rem - lo)) : 0u;
    return ((global_tile_end >> 2) * 2u) + partial;
}

// K-loop extent, in local tiles, of a Q range whose last row is in global tile q_end_global - 1: the local
// tiles visible to that row, but at least one, because reader and compute always process one K chunk per
// Q chunk (a Q range with no visible key, chip 1 at global positions 0..63, is fully masked).
inline uint32_t local_k_end(uint32_t q_end_global, uint32_t c) {
    const uint32_t n = local_tiles_before(q_end_global, c);
    return n != 0 ? n : 1u;
}

// Local causal row position of global Q tile gq: the local K tile holding its diagonal when gq's 32
// positions are on this chip (stamp_diag = true), else the last fully visible local tile (stamp_diag =
// false: no diagonal tile, everything after it is masked). -1 when no local key is visible.
inline int32_t local_q_tile(uint32_t gq, uint32_t c, bool& stamp_diag) {
    stamp_diag = ((gq >> 1) & 1u) == c;
    const int32_t t = static_cast<int32_t>(local_tiles_before(gq, c));
    return stamp_diag ? t : t - 1;
}

// True when every row of global Q tile gq sees no local key (only chip 1, global tiles 0 and 1).
inline bool q_tile_sees_no_key(uint32_t gq, uint32_t c) { return c == 1u && gq < 2u; }

}  // namespace sdpa_pair_split
