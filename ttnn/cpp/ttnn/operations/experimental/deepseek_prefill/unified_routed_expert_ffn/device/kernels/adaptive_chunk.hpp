// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Runtime chunk-size picker, shared by the reader, writer, and compute kernels.
//
// The kernel reads this expert's actual token count device-side and picks
// chunk_M_tiles (hence per_core_M = chunk_M_tiles / GRID_Y and num_chunks)
// FROM THAT COUNT — the adaptive per_core_M work that used to be sized on the
// host from an `expected_tokens` argument now happens at runtime, so no caller
// has to know the token count in advance.
//
// CBs are sized on the host to the compile-time MAX shape (per_core_M = 8,
// chunk_M_tiles = 64; a big-model L1 guard may lower the max, passed in as
// `max_chunk`). The picker never returns more than `max_chunk`, so the runtime
// per_core_M always fits the allocated CBs; a smaller pick simply uses fewer of
// the reserved tiles and shrinks every M-dimension loop.
//
// The three kernels MUST derive identical chunk_M_tiles / per_core_M / chunk
// counts from the same count, or the reader/compute/writer disagree on the row
// mapping and FFN output lands on the wrong token rows. This is the single
// source of truth. Pure integer arithmetic (no NoC/CB/reg APIs) so the header
// is valid in BRISC, NCRISC, and TRISC translation units alike.
namespace adaptive_chunk {

constexpr uint32_t kGridY = 8;  // M-row cores; a chunk spans per_core_M * kGridY tile-rows

// Chunk layout for `count_tiles` tile-rows, given the CB-sized maximum chunk
// `max_chunk` (= per_core_M_max * kGridY): a run of FULL chunks of max_chunk,
// then ONE tail chunk sized down to the remainder. This minimizes the chunk
// count (each chunk re-reads the full gate/up/down weights, the DRAM bottleneck)
// AND minimizes phantom M-work (the tail covers only the leftover rows), so
// per_core_M adapts to the actual token count per chunk with essentially no
// wasted rows — e.g. 160 tiles -> 64 + 64 + 32 (per_core_M 8,8,4), 3 chunks,
// zero phantom.
//
// CRITICAL: the tail per_core_M is a DIVISOR of per_core_M_max. The gate/up CBs
// (cb_in0_x, partials, activated) are allocated to per_core_M_max but reserved/
// pushed at the runtime per_core_M; a per_core_M that does not divide
// per_core_M_max would make those blocks wrap the CB ring (ring size not a
// multiple of the block size) and reserve_back would deadlock. Divisors tile
// evenly, so the ring never wraps mid-block.

// Number of chunks for `count_tiles`: full chunks of max_chunk + one tail chunk.
inline uint32_t num_chunks(uint32_t count_tiles, uint32_t max_chunk) {
    if (count_tiles < 1) {
        return 0;
    }
    const uint32_t num_full = count_tiles / max_chunk;
    const uint32_t tail = count_tiles - num_full * max_chunk;
    return num_full + ((tail > 0) ? 1u : 0u);
}

// per_core_M for chunk index `c`: per_core_M_max for the full chunks; for the
// tail chunk, the smallest DIVISOR of per_core_M_max whose *kGridY covers the
// tail tiles (so the tail does the least M-work while its block still tiles
// evenly into the CBs).
inline uint32_t per_core_M_for_chunk(uint32_t c, uint32_t count_tiles, uint32_t max_chunk) {
    const uint32_t per_core_M_max = max_chunk / kGridY;
    const uint32_t num_full = count_tiles / max_chunk;
    if (c < num_full) {
        return per_core_M_max;
    }
    const uint32_t tail = count_tiles - num_full * max_chunk;  // > 0 for the tail chunk
    uint32_t need = (tail + kGridY - 1) / kGridY;              // min rows/core to cover the tail
    if (need < 1) {
        need = 1;
    }
    for (uint32_t d = need; d <= per_core_M_max; ++d) {
        if ((per_core_M_max % d) == 0) {
            return d;
        }
    }
    return per_core_M_max;
}

}  // namespace adaptive_chunk
