// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Helpers shared by the reader (NCRISC) and writer (BRISC): profiler zones, the NoC drain on return, and for R-QKCORE
// the q / k core window gather, which they split by rows, and the final-state tile order, which pair cores split by
// tiles (GDN_STATE_SPLIT).

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"

// Device profiler zones, compiled in only with the host knob PROFILE_ZONES (GDN_PROFILE_ZONES).
#if GDN_PROFILE_ZONES
#define GDN_ZONE(name) DeviceZoneScopedN(name)
#else
#define GDN_ZONE(name)
#endif
// A point in time (an empty zone), at GDN_PROFILE_ZONES level 2.
#if GDN_PROFILE_ZONES >= 2
#define GDN_MARK(name)  \
    do {                \
        GDN_ZONE(name); \
    } while (0)
#else
#define GDN_MARK(name)
#endif

namespace gdn {

// Declared first in kernel_main so every return drains the kernel's NoC. Semaphore::up(noc, x, y) is a non-posted
// atomic with no barrier of its own, and the next program's firmware seeds its NoC counters from the hardware ones:
// an ack still in flight at return lands after that, and the next kernel's atomic barrier on this NoC never passes.
struct NocDrainOnReturn {
    ~NocDrainOnReturn() { Noc().async_full_barrier(); }
};

// History taps of rows [p_lo, p_hi) of one column: tap j < HISTORY row p = conv_state row HISTORY p + j, from the
// column's cs_tiles staged conv_state tiles into its zeroed tap tiles at col_win. Every row is two face-row segments
// inside this core: one stateful command (own NoC coordinates, segment size), then one address pair per segment.
// Completed by the caller's read barrier.
template <uint32_t TILE_BYTES, uint32_t HISTORY>
FORCE_INLINE void copy_history_rows(uint32_t cs_col, uint32_t col_win, uint32_t p_lo, uint32_t p_hi) {
    constexpr uint32_t TILE_ROWS = 32;
    constexpr uint32_t FACE_BYTES = TILE_BYTES / 4;
    constexpr uint32_t SEG_BYTES = FACE_BYTES / 16;
    auto row_offset = [](uint32_t p) { return (p >> 4) * 2 * FACE_BYTES + (p & 15) * SEG_BYTES; };
    noc_async_read_one_packet_set_state(get_noc_addr(cs_col), SEG_BYTES);
    for (uint32_t p = p_lo; p < p_hi; ++p) {
        for (uint32_t tap = 0; tap < HISTORY; ++tap) {
            const uint32_t r = HISTORY * p + tap;
            const uint32_t src = cs_col + (r / TILE_ROWS) * TILE_BYTES + row_offset(r % TILE_ROWS);
            const uint32_t dst = col_win + tap * TILE_BYTES + row_offset(p);
            noc_async_read_one_packet_with_state(src, dst);
            noc_async_read_one_packet_with_state(src + FACE_BYTES, dst + FACE_BYTES);
        }
    }
}

// recurrent_state_out page of the j-th final-state tile compute packs for the piece whose first page is first_page:
// compute runs the Kt x Vb tiles rotated by (pair * stagger) mod Kt Vb (writer and reader, which split the tiles).
template <uint32_t Kt, uint32_t Vt, uint32_t Vb>
FORCE_INLINE uint32_t state_out_page(uint32_t first_page, uint32_t j, uint32_t stagger) {
    constexpr uint32_t n = Kt * Vb;
    const uint32_t rot = (first_page / (Kt * Vt) * stagger) % n;
    const uint32_t i = j + rot < n ? j + rot : j + rot - n;
    return first_page + (i / Vb) * Vt + i % Vb;
}

// Final-state tile j of a core's unit_idx-th unit is the reader's when the unit is one of its first reader_units, or
// j % share == share - 1 (share 0: none, 1: all). Both are per-core runtime args.
FORCE_INLINE bool state_tile_on_reader(uint32_t unit_idx, uint32_t j, uint32_t reader_units, uint32_t share) {
    return unit_idx < reader_units || (share != 0 && j % share == share - 1);
}

// Rows [0, split) are the reader's, [split, B) the writer's.
FORCE_INLINE uint32_t history_split(uint32_t batch) { return (batch + 1) / 2; }

}  // namespace gdn
