// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Where combine spends the tail: the time after the routed expert has released its last expert, when
// overlapped with it. Each combine kernel times its waits by kind and books each one before the tail, after
// it, or straddling its start; at exit it logs the totals as device-profiler data records, next to the cycle
// it first saw the tail begin.
//
// The tail begins, for a core, when its own `ready` count reaches the last walk step: that is the collector
// saying every routed-expert writer has finished. Seen only when the core looks, so a wait that spans it is
// booked as straddling rather than split.
//
// Built only under the device profiler and only when overlapped (the factory then defines
// CMBF2D_TAIL_READY_SEM and CMBF2D_TAIL_FINAL_STEP); otherwise every call compiles to nothing.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#if defined(PROFILE_KERNEL) && defined(CMBF2D_TAIL_READY_SEM) && defined(CMBF2D_TAIL_FINAL_STEP)
#define CMBF2D_TAIL_PROBE 1
#include "tools/profiler/kernel_profiler.hpp"
#endif

namespace hyb_cmbf2d::tail_probe {

// One bucket set per wait kind; which kinds a kernel uses is its own business.
enum Kind : uint32_t { FWD = 0, UNT, SLOT, LOCAL, READY, RING, FILLED, FABRIC, DRAIN, NUM_KINDS };
enum When : uint32_t { PRE = 0, STRADDLE, TAIL, NUM_WHEN };

#ifdef CMBF2D_TAIL_PROBE

inline uint32_t now() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        RISCV_DEBUG_REG_WALL_CLOCK_L)[kernel_profiler::WALL_CLOCK_LOW_INDEX];
}

struct State {
    uint32_t cycles[NUM_KINDS][NUM_WHEN] = {};
    uint32_t tail_start = 0;
    bool in_tail = false;
};
inline State state;

inline bool look(uint32_t t) {
    if (!state.in_tail) {
        invalidate_l1_cache();
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CMBF2D_TAIL_READY_SEM)) >=
            CMBF2D_TAIL_FINAL_STEP) {
            state.in_tail = true;
            state.tail_start = t;
        }
    }
    return state.in_tail;
}

// Times `body` as a `kind` wait or work region.
template <typename Body>
inline void timed(Kind kind, Body body) {
    const uint32_t t0 = now();
    const bool tail_before = look(t0);
    body();
    const uint32_t t1 = now();
    const bool tail_after = look(t1);
    state.cycles[kind][tail_before ? TAIL : (tail_after ? STRADDLE : PRE)] += t1 - t0;
}

// Logs everything. Record names carry the kind and bucket; the analysis pairs "tail_start" with the
// timestamp of "tail_end" for the tail's length on this RISC.
inline void report() {
    look(now());
    DeviceTimestampedData("cmb_tail_start", state.tail_start);
    DeviceTimestampedData("cmb_tail_seen", static_cast<uint32_t>(state.in_tail));
#define CMBF2D_TAIL_REPORT_KIND(K, NAME)                                            \
    if (state.cycles[K][PRE] | state.cycles[K][STRADDLE] | state.cycles[K][TAIL]) { \
        DeviceTimestampedData(NAME "_pre", state.cycles[K][PRE]);                   \
        DeviceTimestampedData(NAME "_straddle", state.cycles[K][STRADDLE]);         \
        DeviceTimestampedData(NAME "_tail", state.cycles[K][TAIL]);                 \
    }
    CMBF2D_TAIL_REPORT_KIND(FWD, "cmb_w_fwd")
    CMBF2D_TAIL_REPORT_KIND(UNT, "cmb_w_unt")
    CMBF2D_TAIL_REPORT_KIND(SLOT, "cmb_w_slot")
    CMBF2D_TAIL_REPORT_KIND(LOCAL, "cmb_local")
    CMBF2D_TAIL_REPORT_KIND(READY, "cmb_w_ready")
    CMBF2D_TAIL_REPORT_KIND(RING, "cmb_w_ring")
    CMBF2D_TAIL_REPORT_KIND(FILLED, "cmb_w_filled")
    CMBF2D_TAIL_REPORT_KIND(FABRIC, "cmb_w_fabric")
    CMBF2D_TAIL_REPORT_KIND(DRAIN, "cmb_drain")
#undef CMBF2D_TAIL_REPORT_KIND
    DeviceTimestampedData("cmb_tail_end", now());
}

#else

template <typename Body>
inline void timed(Kind, Body body) {
    body();
}
inline void report() {}

#endif

}  // namespace hyb_cmbf2d::tail_probe
