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

// SMALL_INPUTS_MCAST's defines are emitted only by the programs that use it (host time per call grows with the
// defines).
#ifndef GDN_SMALL_MCAST
#define GDN_SMALL_MCAST 0
#define GDN_CB_SMALL_IN 0
#define GDN_SMALL_IN_GATE_PAGES 0
#endif

#ifndef GDN_QK_Q_READER_PCT
#define GDN_QK_Q_READER_PCT 50
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

// Issues the Kt x Vb slice (columns v_first..) of pair `pair`'s S0 into dst, from tile (pair * stagger) mod Kt Vb on so
// that cores starting together spread over the banks; a read barrier after every `per_barrier` tiles but the last
// (the caller completes the rest).
template <uint32_t Kt, uint32_t Vt, uint32_t Vb, uint32_t TILE_BYTES, typename Accessor>
FORCE_INLINE void read_state_slice(
    const Accessor& state, uint32_t dst, uint32_t pair, uint32_t v_first, uint32_t stagger, uint32_t per_barrier) {
    constexpr uint32_t n = Kt * Vb;
    const uint32_t rot = (pair * stagger) % n;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t i = k + rot < n ? k + rot : k + rot - n;
        const uint32_t page = pair * Kt * Vt + (i / Vb) * Vt + v_first + i % Vb;
        noc_async_read(state.get_noc_addr(page), dst + i * TILE_BYTES, TILE_BYTES);
        if ((k + 1) % per_barrier == 0 && k + 1 < n) {
            noc_async_read_barrier();
        }
    }
}

// Final-state tile j of a core's unit_idx-th unit is the reader's when the unit is one of its first reader_units, or
// j % share == share - 1 (share 0: none, 1: all). Both are per-core runtime args.
FORCE_INLINE bool state_tile_on_reader(uint32_t unit_idx, uint32_t j, uint32_t reader_units, uint32_t share) {
    return unit_idx < reader_units || (share != 0 && j % share == share - 1);
}

// Rows [0, split) are the reader's, [split, B) the writer's. A q core's writer also sends the small inputs
// (SMALL_INPUTS_MCAST), so its reader takes GDN_QK_Q_READER_PCT percent of the rows (host QK_Q_READER_PCT).
FORCE_INLINE uint32_t history_split(uint32_t batch, uint32_t reader_pct = 50) {
    return (batch * reader_pct + 50) / 100;
}

}  // namespace gdn
