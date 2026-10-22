// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Regime-R3 combine transport shared by the reader (partials, reader-side all-gather) and the writer (writer-side
// all-gather) — Perf 1, perf_experiments/combine_bytes.
//
// combine_write_tiles: one R3 combine tile (a REDUCE_ROW partial or a finalized inv_rms tile, Float16_b / Float32 —
// never Bfp8_b, so a face is exactly T/4 bytes) crosses the NoC as FACES 0 AND 2 ONLY: half the bytes of the whole
// tile, as one-packet stateful writes (the destination core is programmed once per call). Faces 1/3 are never read:
// every combine tile has its valid data in column 0 and every consumer is lane-wise or reads faces 0/2 only —
//   combine  = reduce<SUM, REDUCE_COL, AccumulateViaAdd, ReduceWithinTile::Skip> (pairwise add_tiles: lane-wise),
//   finalize = lane-wise SFPU (FINALIZE_COL0: the even vectors of faces 0/2; stock: full-tile lane-wise),
//   normalize = BroadcastDim::Col, whose srcB unpack (llk_unpack_AB, COL) loads face 0 and face 2 only.
// So the receiving pages' faces 1/3 may hold anything (stale data, NaN): measured bit-exact with NaN-poisoned
// faces 1/3 on 14 geometries; a face-0-only negative control fails (the poison does reach the consumers).
// Measured (8192x1024 BLOCK [1024,128] 8x8, one-shot combine): 22.7 -> 18.6 us; + rotation 18.2 us.
// Raw NoC: no kernel_lib helper writes a face-strided subset of a tile (the combine transport was raw NoC already).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace rms_norm_combine {

template <uint32_t T>
FORCE_INLINE void combine_write_tiles(uint32_t src, uint32_t x, uint32_t y, uint32_t dst_l1, uint32_t n) {
    constexpr uint32_t F = T / 4;  // one 16x16 face
    static_assert(F <= NOC_MAX_BURST_SIZE, "one face must fit one NoC packet");
    if (n == 0) {
        return;
    }
    noc_async_write_one_packet_set_state(get_noc_addr(x, y, dst_l1), F);
    for (uint32_t t = 0; t < n; ++t) {
        noc_async_write_one_packet_with_state(src + t * T, dst_l1 + t * T);
        noc_async_write_one_packet_with_state(src + t * T + 2 * F, dst_l1 + t * T + 2 * F);
    }
}

// Rotated destination order: member `my_index` walks destinations starting at the next one, so the row-group's
// senders do not all queue on destination 0 first (measured 22.7 -> 21.3 us alone at whole tiles on 8192x1024 BLOCK).
FORCE_INLINE uint32_t rotated(uint32_t i, uint32_t my_index, uint32_t n) {
    uint32_t j = my_index + 1 + i;
    while (j >= n) {
        j -= n;
    }
    return j;
}

}  // namespace rms_norm_combine
