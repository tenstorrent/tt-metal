// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa row-sum service (runs on the writer core, which is otherwise idle).
//
// The compute kernel cannot accumulate the online-softmax row sum exactly: DEST is 16-bit and the
// packer's L1 accumulation adds in that precision, so a running sum that grows to hundreds absorbs
// the small per-visit increments (measured: -11% at ~100 visits, -40% at ~400). Instead the compute
// reduces each visit's probs to per-row PARTIAL sums in the FPU (fp32 internally, one bf16 rounding)
// and streams them here; this service accumulates them in 64-bit fixed point (exact), applies the
// rare anchor-move correction factors, and hands the row sums back as a bf16 tile at flush.
//
// Header page (16 B) on cb_hdr: {kind, row_slot, sqt, 0}; tiles follow on cb_tiles (bf16, column 0
// meaningful, rows face-major):
//   kind 0 PARTIAL  sqt tiles: row sums of this visit (row's first visit resets the accumulator)
//   kind 1 CORR     sqt tiles: multiply the accumulator by column 0 (anchor moved)
//   kind 2 FLUSH    no tiles: reply with sqt tiles of bf16 sums on cb_sumback
//   kind 3 FIRST    no tiles: reset the row (sent before a row's first partial)
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace vsa_sum {

constexpr uint32_t KIND_PARTIAL = 0;
constexpr uint32_t KIND_CORR = 1;
constexpr uint32_t KIND_FLUSH = 2;
constexpr uint32_t KIND_FIRST = 3;
constexpr int kScale = 16;  // fixed-point scale 2^16; the 64-bit accumulator (hi:lo) takes ~2^47 of value

// column-0 element of tile row r (bf16 units; 16x16 faces, face-major)
FORCE_INLINE uint32_t col0_off(uint32_t r) { return (r < 16 ? 0u : 512u) + (r & 15) * 16; }

// acc(hi:lo) += bf16 value b (>= 0) in fixed point 2^kScale (64-bit safe for any bf16 magnitude)
FORCE_INLINE void add_bf16(volatile tt_l1_ptr uint32_t* lo, volatile tt_l1_ptr uint32_t* hi, uint16_t b) {
    const uint32_t e = (b >> 7) & 0xFF;
    if (e < 134 - kScale || (b & 0x8000)) {  // < 2^-16 (or negative: cannot happen): nothing to add
        return;
    }
    const uint32_t m = 0x80u | (b & 0x7F);
    const int sh = static_cast<int>(e) - 134 + kScale;  // value = m * 2^(e-134)
    uint32_t add_lo, add_hi = 0;
    if (sh < 0) {
        add_lo = m >> (-sh);
    } else if (sh <= 24) {
        add_lo = m << sh;
    } else {
        const uint64_t v = static_cast<uint64_t>(m) << sh;
        add_lo = static_cast<uint32_t>(v);
        add_hi = static_cast<uint32_t>(v >> 32);
    }
    const uint32_t old = *lo;
    const uint32_t nw = old + add_lo;
    *lo = nw;
    *hi += add_hi + (nw < old ? 1u : 0u);
}

// acc(hi:lo) * corr for corr = bf16 in [0, 1]  (rare: only when a row's anchor moves)
FORCE_INLINE void mul_bf16(volatile tt_l1_ptr uint32_t* lo, volatile tt_l1_ptr uint32_t* hi, uint16_t b) {
    const uint32_t e = (b >> 7) & 0xFF;
    const uint64_t acc = (static_cast<uint64_t>(*hi) << 32) | *lo;
    if (e == 0 || acc == 0) {
        *lo = 0;
        *hi = 0;
        return;
    }
    const uint64_t m = 0x80u | (b & 0x7F);
    const int sh = 134 - static_cast<int>(e);  // corr = m * 2^-sh
    const uint64_t p = acc * m;
    const uint64_t r = sh >= 0 ? (p >> sh) : (p << (-sh));
    *lo = static_cast<uint32_t>(r);
    *hi = static_cast<uint32_t>(r >> 32);
}

FORCE_INLINE uint16_t fixed_to_bf16(uint32_t lo, uint32_t hi) {
    const uint64_t v = (static_cast<uint64_t>(hi) << 32) | lo;
    if (v == 0) {
        return 0;
    }
    const int msb = 63 - __builtin_clzll(static_cast<unsigned long long>(v));
    int sh = msb - 7;  // round to nearest on the 7 kept mantissa bits
    uint64_t mant = sh > 0 ? ((v + (1ull << (sh - 1))) >> sh) : (v << (-sh));
    int e = msb - kScale + 127;
    if (mant & 0x100) {
        mant >>= 1;
        ++e;
    }
    return static_cast<uint16_t>((e << 7) | (mant & 0x7F));
}

struct Service {
    uint32_t cb_hdr, cb_tiles, cb_sumback, acc_l1;  // acc: uint32 lo[64], hi[64] per row in an L1 scratch CB
    uint32_t n_rows;
    uint32_t sqt;
    uint32_t zeroed_pages = 0;
    volatile tt_l1_ptr uint32_t* acc_lo(uint32_t row) const {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(acc_l1) + row * 128;
    }
    volatile tt_l1_ptr uint32_t* acc_hi(uint32_t row) const { return acc_lo(row) + 64; }

    // Resumable: at most kRowsPerCall accumulator rows per call, so the caller's K-pull service (which the
    // reader's emission waits on) is never delayed by more than a few hundred cycles. Never blocks on
    // something the compute has not produced yet.
    static constexpr uint32_t kRowsPerCall = 16;
    uint32_t cur_kind = 0xFFFFFFFFu, cur_row = 0, cur_ntiles = 0, cur_pos = 0;  // in-progress tile item

    void serve() {
        if (cur_kind == 0xFFFFFFFFu) {
            if (!cb_pages_available_at_front(cb_hdr, 1)) {
                return;
            }
            cb_wait_front(cb_hdr, 1);
            volatile tt_l1_ptr uint32_t* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_hdr));
            invalidate_l1_cache();
            const uint32_t kind = h[0], row = h[1], ntiles = h[2];
            cb_pop_front(cb_hdr, 1);
            if (kind == KIND_FIRST) {
                volatile tt_l1_ptr uint32_t* lo = acc_lo(row);
                for (uint32_t r = 0; r < 128; ++r) {  // lo[64] then hi[64]
                    lo[r] = 0;
                }
                return;
            }
            if (kind == KIND_FLUSH) {
                volatile tt_l1_ptr uint32_t* lo = acc_lo(row);
                volatile tt_l1_ptr uint32_t* hi = acc_hi(row);
                cb_reserve_back(cb_sumback, sqt);
                volatile tt_l1_ptr uint16_t* out =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_sumback));
                if (zeroed_pages < 2) {  // only column 0 is read downstream; zero each ring half once per launch
                    for (uint32_t i = 0; i < sqt * 1024; ++i) {
                        out[i] = 0;
                    }
                    ++zeroed_pages;
                }
                for (uint32_t t = 0; t < sqt; ++t) {
                    volatile tt_l1_ptr uint16_t* tile = out + t * 1024;
                    for (uint32_t r = 0; r < 32; ++r) {
                        tile[col0_off(r)] = fixed_to_bf16(lo[t * 32 + r], hi[t * 32 + r]);
                    }
                }
                cb_push_back(cb_sumback, sqt);
                return;
            }
            // PARTIAL / CORR: tiles follow; process in slices
            cb_wait_front(cb_tiles, ntiles);
            invalidate_l1_cache();
            cur_kind = kind;
            cur_row = row;
            cur_ntiles = ntiles;
            cur_pos = 0;
        }
        // one slice of the in-progress item. The ring slot was rewritten by the compute's packer since this
        // RISC last read the same addresses: drop the L1 read cache first (stale partials/corrs otherwise).
        invalidate_l1_cache();
        const tt_l1_ptr uint16_t* tiles = reinterpret_cast<const tt_l1_ptr uint16_t*>(get_read_ptr(cb_tiles));
        volatile tt_l1_ptr uint32_t* lo = acc_lo(cur_row);
        volatile tt_l1_ptr uint32_t* hi = acc_hi(cur_row);
        const uint32_t end = cur_pos + kRowsPerCall;  // positions index rows across the item's tiles
        for (uint32_t p = cur_pos; p < end; ++p) {
            const uint32_t t = p >> 5, r = p & 31;
            const uint16_t b = tiles[t * 1024 + col0_off(r)];
            if (cur_kind == KIND_PARTIAL) {
                add_bf16(&lo[p], &hi[p], b);
            } else {
                mul_bf16(&lo[p], &hi[p], b);
            }
        }
        cur_pos = end;
        if (cur_pos >= cur_ntiles * 32) {
            cb_pop_front(cb_tiles, cur_ntiles);
            cur_kind = 0xFFFFFFFFu;
        }
    }
    // true while an item is mid-flight (callers may want to finish it before blocking elsewhere)
    bool busy() const { return cur_kind != 0xFFFFFFFFu; }
};

}  // namespace vsa_sum
