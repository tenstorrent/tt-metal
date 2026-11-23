// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Banded relative bias (SDPA rel_logits) helpers shared by the reader and, with SDPA_REL_WRITER, the writer (the
// RISC that builds the skewed bias blocks). Include only with SDPA_REL_BIAS defined.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

constexpr uint16_t kRelMasked = 0xCE6E;  // bf16 -1.0e9
constexpr uint32_t kRelW = SDPA_REL_W;
constexpr uint32_t kRelQOff = SDPA_REL_Q_OFF;
constexpr uint32_t kRelRowBytes = SDPA_REL_ROW_BYTES;

#ifdef SDPA_REL_TILIZE
// row[c] = v for c in [c0, c1) of a 4 B aligned bf16 row: 32-bit stores, a 16-bit store at an odd edge.
FORCE_INLINE void rel_fill_run(tt_l1_ptr uint16_t* row, int32_t c0, int32_t c1, uint16_t v) {
    if (c0 & 1) {
        row[c0++] = v;
    }
    if (c1 > c0 && (c1 & 1)) {
        row[--c1] = v;
    }
    tt_l1_ptr uint32_t* w = reinterpret_cast<tt_l1_ptr uint32_t*>(row);
    const uint32_t vv = uint32_t{v} << 16 | v;
    for (int32_t i = c0 >> 1; i < (c1 >> 1); ++i) {
        w[i] = vv;
    }
}

// Skewed layout, compute tilizes: one row-major [32, kc] bias block for global Q tile row q_tile and the K chunk at
// key k0. Row r's bias for keys k0 .. sits at columns C0 .. of its skewed row (C0 = W - q_tile * 32 - off + k0, the
// same for every row, 64 B aligned), so each row is one DMA; then only the out-of-band entries are stored: d < 0
// and keys past the valid ones masked, d >= W zero (masked past a sliding window).
// The out-of-band entries of a row-major bias block whose in-band bias bytes have landed (see rel_rm_block).
template <uint32_t kc, uint32_t window>
FORCE_INLINE void rel_rm_fill(uint32_t q_tile, uint32_t k0, uint32_t valid_k, bool q_valid, uint32_t blk) {
    constexpr int32_t W = static_cast<int32_t>(kRelW);
    volatile tt_l1_ptr uint16_t* b = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(blk);
    if (!q_valid) {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blk);
        for (uint32_t i = 0; i < 32 * kc / 2; ++i) {
            w[i] = uint32_t{kRelMasked} << 16 | kRelMasked;
        }
        return;
    }
    asm volatile("" ::: "memory");
    for (uint32_t r = 0; r < 32; ++r) {
        const int32_t p = static_cast<int32_t>(q_tile * 32 + r + kRelQOff) - static_cast<int32_t>(k0);  // d at c = 0
        tt_l1_ptr uint16_t* row = const_cast<tt_l1_ptr uint16_t*>(b + r * kc);
        // c > p: d < 0, masked (also keys past valid_k)
        int32_t c_mask = p + 1 < 0 ? 0 : p + 1;
        const int32_t c_valid = static_cast<int32_t>(valid_k) - static_cast<int32_t>(k0);
        if (c_valid < c_mask) {
            c_mask = c_valid < 0 ? 0 : c_valid;
        }
        if (c_mask < static_cast<int32_t>(kc)) {
            rel_fill_run(row, c_mask, kc, kRelMasked);
        }
        // c <= p - W: d >= W, no bias (or masked past the window)
        const int32_t c_far = p - W + 1;  // columns [0, c_far)
        const int32_t c_end = c_far > c_mask ? c_mask : c_far;
        if (c_end > 0) {
            if constexpr (window > 0 && window <= static_cast<uint32_t>(W)) {
                rel_fill_run(row, 0, c_end, kRelMasked);  // d >= W >= window: all masked
            } else if constexpr (window == 0) {
                rel_fill_run(row, 0, c_end, 0);  // d >= W: no bias
            } else {
                for (int32_t c = 0; c < c_end; ++c) {
                    row[c] = (window > 0 && p - c >= static_cast<int32_t>(window)) ? kRelMasked : uint16_t{0};
                }
            }
        }
    }
    asm volatile("" ::: "memory");
}

template <uint32_t kc, uint32_t window, typename Reader>
FORCE_INLINE void rel_rm_block(
    Noc& noc,
    const Reader& rel_reader,
    uint32_t rel_page0,
    uint32_t q_tile,
    uint32_t k0,
    uint32_t valid_k,
    bool q_valid,
    uint32_t blk) {
    constexpr int32_t W = static_cast<int32_t>(kRelW);
    constexpr int32_t row_cols = W + 32;
    if (!q_valid) {
        rel_rm_fill<kc, window>(q_tile, k0, valid_k, false, blk);
        return;
    }
    const int32_t C0 = W - static_cast<int32_t>(q_tile * 32 + kRelQOff) + static_cast<int32_t>(k0);
    const int32_t y_lo = C0 < 0 ? 0 : C0;
    const int32_t y_hi = C0 + static_cast<int32_t>(kc) > row_cols ? row_cols : C0 + static_cast<int32_t>(kc);
    if (y_lo < y_hi) {
        for (uint32_t r = 0; r < 32; ++r) {
            noc.async_read(
                rel_reader,
                CoreLocalMem<uint32_t>(blk + r * kc * 2 + (y_lo - C0) * 2),
                (y_hi - y_lo) * 2,
                {.page_id = rel_page0 + q_tile * 32 + r, .offset_bytes = static_cast<uint32_t>(y_lo) * 2},
                {});
        }
        noc.async_read_barrier();
    }
    rel_rm_fill<kc, window>(q_tile, k0, valid_k, true, blk);
}

#ifdef SDPA_REL_STAGE_CB
// Staged bias rows (writer): per group of G K chunks, each bias row of a Q chunk is read once, as one run of up to
// G * kc elements, into an L1 staging buffer; the blocks of the group's chunks are then copied out of it (local
// NOC reads). Same bytes as rel_rm_block's per-block row reads, in reads G times longer.
struct RelRange {
    uint32_t k_near;  // first K chunk with a bias block (chunks entirely >= W behind the first query have none)
    uint32_t k_hi;
};
template <uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t window>
FORCE_INLINE RelRange rel_range(uint32_t q_chunk, uint32_t k_num_chunks) {
    constexpr uint32_t kc = Sk_chunk_t * 32;
    const uint32_t p0 = q_chunk * Sq_chunk_t * 32 + kRelQOff;
    const uint32_t p1 = p0 + Sq_chunk_t * 32 - 1;
    uint32_t k = (window > 0 && p0 + 1 > window) ? (p0 + 1 - window) / kc : 0;
    uint32_t k_hi = p1 / kc + 1;
    k_hi = k_hi > k_num_chunks ? k_num_chunks : k_hi;
    while (k < k_hi &&
           static_cast<int32_t>(p0) - static_cast<int32_t>((k + 1) * kc - 1) >= static_cast<int32_t>(kRelW)) {
        ++k;
    }
    return {k, k_hi};
}

// Issue (no barrier) the staging reads of the group starting at K chunk kg of Q chunk q_chunk into buf: row r of
// Q tile row j at buf + (32 j + r) * G * kc * 2, element y - C0g at offset 2 (y - C0g), C0g = the group's first
// column (rel_rm_block's C0 of chunk kg).
template <uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t G, typename Reader>
FORCE_INLINE void rel_stage_issue(
    const Reader& rel_reader, uint32_t rel_page0, uint32_t q_chunk, uint32_t kg, uint32_t q_tiles, uint32_t buf) {
    constexpr int32_t W = static_cast<int32_t>(kRelW);
    constexpr int32_t row_cols = W + 32;
    constexpr uint32_t kc = Sk_chunk_t * 32, row_b = G * kc * 2;
    for (uint32_t j = 0; j < Sq_chunk_t; ++j) {
        const uint32_t t = q_chunk * Sq_chunk_t + j;
        if (t >= q_tiles) {
            continue;
        }
        const int32_t C0g = W - static_cast<int32_t>(t * 32 + kRelQOff) + static_cast<int32_t>(kg * kc);
        const int32_t y0 = C0g < 0 ? 0 : C0g;
        const int32_t y1 =
            C0g + static_cast<int32_t>(G * kc) > row_cols ? row_cols : C0g + static_cast<int32_t>(G * kc);
        if (y0 >= y1) {
            continue;
        }
        const uint32_t off = static_cast<uint32_t>(y0) * 2, bytes = static_cast<uint32_t>(y1 - y0) * 2;
        const uint32_t dst = buf + j * 32 * row_b + static_cast<uint32_t>(y0 - C0g) * 2;
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read(rel_reader.get_noc_addr(rel_page0 + t * 32 + r, off), dst + r * row_b, bytes);
        }
    }
}

// Issue (no barrier) the copies of block (Q tile row j, K chunk k) out of the staged group starting at chunk kg.
template <uint32_t Sq_chunk_t, uint32_t Sk_chunk_t, uint32_t G>
FORCE_INLINE void rel_stage_block(uint32_t buf, uint32_t q_chunk, uint32_t j, uint32_t kg, uint32_t k, uint32_t blk) {
    constexpr int32_t W = static_cast<int32_t>(kRelW);
    constexpr int32_t row_cols = W + 32;
    constexpr uint32_t kc = Sk_chunk_t * 32, row_b = G * kc * 2;
    const uint32_t t = q_chunk * Sq_chunk_t + j;
    const int32_t C0 = W - static_cast<int32_t>(t * 32 + kRelQOff) + static_cast<int32_t>(k * kc);
    const int32_t C0g = C0 - static_cast<int32_t>((k - kg) * kc);
    const int32_t y_lo = C0 < 0 ? 0 : C0;
    const int32_t y_hi = C0 + static_cast<int32_t>(kc) > row_cols ? row_cols : C0 + static_cast<int32_t>(kc);
    if (y_lo >= y_hi) {
        return;
    }
    const uint32_t src = buf + j * 32 * row_b + static_cast<uint32_t>(y_lo - C0g) * 2;
    const uint32_t dst = blk + static_cast<uint32_t>(y_lo - C0) * 2, bytes = static_cast<uint32_t>(y_hi - y_lo) * 2;
    for (uint32_t r = 0; r < 32; ++r) {
        noc_async_read(get_noc_addr(src + r * row_b), dst + r * kc * 2, bytes);
    }
}
#endif
#endif
