// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast R2 (`rm_pseudo_tile_stream`) row movers shared by the reader (compact) and the writer (expand).
//
// A block of `rows` interleaved RM rows sits in a staging zone in bank-run order: run j (j = 0..nb-1) holds
// rows j, j+nb, j+2nb, ... at the aligned-page stride `slot_bytes`, so dense row i (j = i % nb, k = i / nb)
// lives in slot  j*q + min(j, r) + k  (q = rows / nb, r = rows % nb). The dense side packs rows back to back
// (`row_bytes` each) as 1024-value pseudo-tiles.
//
// The movers walk the DENSE order so the dense side is a contiguous word stream: groups of G rows
// (G * row_bytes a multiple of 4) are funnelled through a 64-bit accumulator with compile-time shifts —
// whole-word loads/stores only, instead of one byte/halfword load+store per byte (Refinement 3: the per-row
// RISC-V copy was the binding stage of F322). Rows whose size is already a multiple of 4 are plain word
// copies; large odd-sized rows (no profitable unroll) and the < G leftover rows of a block use the
// byte/halfword copy. Writer-side slot words may carry bytes of the next row past `row_bytes`: they land in
// the aligned page's padding (don't-care); slots are >= roundup(row_bytes, 4).

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_rm {

constexpr uint32_t gcd(uint32_t a, uint32_t b) { return b == 0 ? a : gcd(b, a % b); }

// Funnel unroll budget: groups whose total source words exceed this fall back to the per-row copy.
constexpr uint32_t kMaxGroupWords = 16;

template <uint32_t row_bytes>
struct RowGeom {
    static constexpr uint32_t group_rows = 4 / gcd(row_bytes, 4);  // G
    static constexpr uint32_t row_words = (row_bytes + 3) / 4;     // words a row touches in its slot
    static constexpr uint32_t group_dense_words = group_rows * row_bytes / 4;
    static constexpr bool word_rows = (row_bytes % 4) == 0;
    static constexpr bool funnel = !word_rows && group_rows * row_words <= kMaxGroupWords;
};

// Byte/halfword/word copy of one row (unaligned-tolerant fallback, tails).
template <uint32_t row_bytes>
FORCE_INLINE void copy_row(uint32_t src, uint32_t dst) {
    if constexpr (row_bytes % 4 == 0) {
        auto* s = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(src);
        auto* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        for (uint32_t i = 0; i < row_bytes / 4; ++i) {
            d[i] = s[i];
        }
    } else if constexpr (row_bytes % 2 == 0) {
        auto* s = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(src);
        auto* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
        for (uint32_t i = 0; i < row_bytes / 2; ++i) {
            d[i] = s[i];
        }
    } else {
        auto* s = reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(src);
        auto* d = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst);
        for (uint32_t i = 0; i < row_bytes; ++i) {
            d[i] = s[i];
        }
    }
}

// Staging slot address of successive dense rows (bank-run order, see file header).
struct SlotWalker {
    uint32_t run_base;  // staging + prefix(j) * slot_bytes
    uint32_t k_off;     // k * slot_bytes
    uint32_t j;
    uint32_t staging, slot_bytes, nb, q, r;

    FORCE_INLINE SlotWalker(uint32_t staging_, uint32_t slot_bytes_, uint32_t nb_, uint32_t rows) :
        run_base(staging_), k_off(0), j(0), staging(staging_), slot_bytes(slot_bytes_), nb(nb_) {
        q = rows / nb_;
        r = rows - q * nb_;
    }
    FORCE_INLINE uint32_t addr() const { return run_base + k_off; }
    FORCE_INLINE void next() {
        run_base += (q + (j < r ? 1 : 0)) * slot_bytes;
        if (++j == nb) {
            j = 0;
            run_base = staging;
            k_off += slot_bytes;
        }
    }
};

// Reader: staging (bank-run order, aligned slots) -> dense rows at `dense`.
template <uint32_t row_bytes>
FORCE_INLINE void compact_block(uint32_t staging, uint32_t slot_bytes, uint32_t nb, uint32_t rows, uint32_t dense) {
    using G = RowGeom<row_bytes>;
    SlotWalker w(staging, slot_bytes, nb, rows);
    uint32_t i = 0;
    if constexpr (G::word_rows) {
        auto* d = reinterpret_cast<tt_l1_ptr uint32_t*>(dense);
        for (; i < rows; ++i) {
            const auto* s = reinterpret_cast<const tt_l1_ptr uint32_t*>(w.addr());
#pragma GCC unroll 8
            for (uint32_t x = 0; x < G::row_words; ++x) {
                d[x] = s[x];
            }
            d += G::row_words;
            w.next();
        }
    } else if constexpr (G::funnel) {
        auto* d = reinterpret_cast<tt_l1_ptr uint32_t*>(dense);
        const uint32_t full = rows - rows % G::group_rows;
        for (; i < full; i += G::group_rows) {
            uint32_t src[G::group_rows];
#pragma GCC unroll 16
            for (uint32_t g = 0; g < G::group_rows; ++g) {
                src[g] = w.addr();
                w.next();
            }
            uint64_t acc = 0;
            uint32_t nbits = 0;
            uint32_t o = 0;
#pragma GCC unroll 16
            for (uint32_t g = 0; g < G::group_rows; ++g) {
                const auto* s = reinterpret_cast<const tt_l1_ptr uint32_t*>(src[g]);
#pragma GCC unroll 16
                for (uint32_t x = 0; x < G::row_words; ++x) {
                    const uint32_t valid = (x + 1 < G::row_words) ? 4 : row_bytes - 4 * x;
                    uint32_t v = s[x];
                    if (valid < 4) {
                        v &= (1u << (8 * valid)) - 1;
                    }
                    acc |= static_cast<uint64_t>(v) << nbits;
                    nbits += 8 * valid;
                    if (nbits >= 32) {
                        d[o++] = static_cast<uint32_t>(acc);
                        acc >>= 32;
                        nbits -= 32;
                    }
                }
            }
            d += G::group_dense_words;
        }
    }
    for (; i < rows; ++i) {
        copy_row<row_bytes>(w.addr(), dense + i * row_bytes);
        w.next();
    }
}

// Writer: dense rows at `dense` -> staging (bank-run order, aligned slots).
template <uint32_t row_bytes>
FORCE_INLINE void expand_block(uint32_t dense, uint32_t staging, uint32_t slot_bytes, uint32_t nb, uint32_t rows) {
    using G = RowGeom<row_bytes>;
    SlotWalker w(staging, slot_bytes, nb, rows);
    uint32_t i = 0;
    if constexpr (G::word_rows) {
        const auto* s = reinterpret_cast<const tt_l1_ptr uint32_t*>(dense);
        for (; i < rows; ++i) {
            auto* d = reinterpret_cast<tt_l1_ptr uint32_t*>(w.addr());
#pragma GCC unroll 8
            for (uint32_t x = 0; x < G::row_words; ++x) {
                d[x] = s[x];
            }
            s += G::row_words;
            w.next();
        }
    } else if constexpr (G::funnel) {
        const auto* s = reinterpret_cast<const tt_l1_ptr uint32_t*>(dense);
        const uint32_t full = rows - rows % G::group_rows;
        for (; i < full; i += G::group_rows) {
            uint32_t in[G::group_dense_words + 1];
#pragma GCC unroll 16
            for (uint32_t x = 0; x < G::group_dense_words; ++x) {
                in[x] = s[x];
            }
            in[G::group_dense_words] = 0;
            // Bit position p of the group stream = word p/32, bit p%32 (compile-time after unrolling).
#pragma GCC unroll 16
            for (uint32_t g = 0; g < G::group_rows; ++g) {
                auto* d = reinterpret_cast<tt_l1_ptr uint32_t*>(w.addr());
#pragma GCC unroll 16
                for (uint32_t x = 0; x < G::row_words; ++x) {
                    const uint32_t p = g * row_bytes * 8 + x * 32;
                    const uint32_t wi = p / 32;
                    const uint32_t sh = p % 32;
                    d[x] = sh == 0 ? in[wi] : (in[wi] >> sh) | (in[wi + 1] << (32 - sh));
                }
                w.next();
            }
            s += G::group_dense_words;
        }
    }
    for (; i < rows; ++i) {
        copy_row<row_bytes>(dense + i * row_bytes, w.addr());
        w.next();
    }
}

}  // namespace g6_rm
