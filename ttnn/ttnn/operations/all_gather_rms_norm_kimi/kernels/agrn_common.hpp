// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// AGRN_ZONES (host define): device-profiler zones around every phase; off by default (they overflow the profiler
// buffer in long traced runs).
#if defined(AGRN_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define AGRN_ZONE(name) DeviceZoneScopedN(name)
#else
#define AGRN_ZONE(name)
#endif

// CB indices shared by the host program and every kernel of all_gather_rms_norm_kimi.
namespace agrn {
constexpr uint32_t CB_LAND = 0;    // bf16: the row block, both call parities (persistent L1 landing buffer)
constexpr uint32_t CB_GAMMA = 1;   // bf16: gamma tiles of this core's columns (row 0 of each tile valid)
constexpr uint32_t CB_SCALER = 2;  // bf16: reduce scaler (1.0)
constexpr uint32_t CB_SQ = 3;      // (unused)
constexpr uint32_t CB_PART = 4;    // fp32: S = sum_j x_j^2 of my block row (exchanged)
constexpr uint32_t CB_GATHER = 5;  // fp32: the G S-tiles of this row, slot p = slice p
constexpr uint32_t CB_INV = 6;     // fp32: 1 / rms (column 0 valid)
constexpr uint32_t CB_XG = 7;      // fp32: x * gamma
constexpr uint32_t CB_OUT = 8;     // bf16: output tiles
constexpr uint32_t CB_STAGE = 9;  // port cores: one slot per packet the port sends (payload + tag), filled by row cores
}  // namespace agrn

namespace agrn {
constexpr uint32_t TAG_BYTES = 16;

constexpr uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Landing layout of one tile-row of a slice: the row's Wt tiles travel as packets (columns [0, a) and [a, Wt), each
// cut into packets of at most P tiles); every packet is followed by one pad slot whose first 16 B hold the packet's
// tag (the call counter + 1), written last. A receiver polls the tag: no arrival counter, no router-side increment.
template <uint32_t Wt, uint32_t a, uint32_t P>
struct RowLayout {
    static constexpr uint32_t ka = cdiv(a, P);
    static constexpr uint32_t npk = ka + cdiv(Wt - a, P);
    static constexpr uint32_t slots = Wt + npk;  // tile slots per row
    static constexpr uint32_t slot_of(uint32_t j) { return j < a ? j + j / P : j + ka + (j - a) / P; }
    // packet k: first tile, tile count
    static constexpr uint32_t first(uint32_t k) { return k < ka ? k * P : a + (k - ka) * P; }
    static constexpr uint32_t count(uint32_t k) {
        const uint32_t f = first(k);
        const uint32_t end = k < ka ? a : Wt;
        return (end - f) < P ? (end - f) : P;
    }
};

#ifndef COMPILE_FOR_TRISC
// The tag is 4 words: (call counter + 1, ~that, that ^ salt, salt), salt = a random per-op-state word (a runtime arg).
// Staging slots are program CB space other ops use between calls, and L1 keeps its contents across processes, where the
// call counter restarts: without the salt, a leftover of an earlier process's call with the same counter would match.
FORCE_INLINE bool tag_ok(uint32_t addr, uint32_t tag, uint32_t salt) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    return p[0] == tag && p[1] == ~tag && p[2] == (tag ^ salt) && p[3] == salt;
}

FORCE_INLINE void wait_tag(uint32_t addr, uint32_t tag, uint32_t salt) {
    do {
        invalidate_l1_cache();
    } while (!tag_ok(addr, tag, salt));
}

FORCE_INLINE void write_tag(uint32_t addr, uint32_t tag, uint32_t salt) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    p[0] = tag;
    p[1] = ~tag;
    p[2] = tag ^ salt;
    p[3] = salt;
}
#endif
}  // namespace agrn
