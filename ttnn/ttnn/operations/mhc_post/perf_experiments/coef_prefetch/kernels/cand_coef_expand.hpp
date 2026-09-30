// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// coef_prefetch candidate — load_coefficients (dataflow only). Same layout and bit-for-bit copies as the op's
// mhc_post_coef_expand.hpp (term t of stream j in half t%2 of tile j*P + t/2; term 0 = post(rho, j),
// term 1+i = comb(rho, i*n + j)). Differences, each behind a compile-time switch:
//   FAST  — expand_half software-pipelines the raw load of face row r+1 behind the 16 stores of row r
//           (the op's loop stalls on the load-use of every row: ~1.75 cycles per stored word measured).
//   SHARE — the writer (BRISC, sole CB-API producer of cb_coef_raw / cb_coef_bcast) expands only the terms of
//           even global index g = j*(n+1) + t; the reader (NCRISC, ReaderShare below) expands the odd ones into
//           the window the writer reserved, in its reserve / barrier shadow. Hand-off: two L1 semaphores holding
//           monotonic counters — exp_go (writer -> reader: set s reserved + raw landed, value s+1) and exp_done
//           (reader -> writer: reader's share of stream j of set s written, value s*n + j + 1). The writer pushes
//           stream j's P tiles only when both shares of stream j are written; it pops cb_coef_raw (and may start
//           the next set over it) only after the reader signalled all n streams.
//           The window address is derived, not exchanged: every reserve of cb_coef_bcast is one full set and the
//           CB is exactly COEF_DEPTH sets, so set s lives at fifo_base + (s % COEF_DEPTH) * set_bytes; cb_coef_raw
//           is exactly one set, so the raw tiles always sit at its fifo base.
// No raw LLK: dataflow_api only.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_post_common.hpp"

namespace mhc_post {

constexpr uint32_t FACE_HW = 16;
constexpr uint32_t FACE_ELEMS = FACE_HW * FACE_HW;

FORCE_INLINE uint32_t cp_sem_read(uint32_t addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}
FORCE_INLINE void cp_sem_write(uint32_t addr, uint32_t v) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = v; }

// The op's expansion loop, verbatim.
FORCE_INLINE void expand_half_ref(uint32_t raw_tile_addr, uint32_t col, uint32_t dst_tile_addr, uint32_t half) {
#pragma GCC unroll 1
    for (uint32_t fh = 0; fh < 2; ++fh) {
        const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_tile_addr) +
                                                 (2 * fh + col / FACE_HW) * FACE_ELEMS + (col % FACE_HW);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile_addr) + (2 * fh + half) * FACE_ELEMS;
#pragma GCC unroll 1
        for (uint32_t r = 0; r < FACE_HW; ++r) {
            const uint32_t bits = src[r * FACE_HW];
#pragma GCC unroll 16
            for (uint32_t w = 0; w < FACE_HW; ++w) {
                dst[w] = bits;
            }
            dst += FACE_HW;
        }
    }
}

// Software-pipelined: the load of row r+1 is issued before the 16 stores of row r.
FORCE_INLINE void expand_half_fast(uint32_t raw_tile_addr, uint32_t col, uint32_t dst_tile_addr, uint32_t half) {
#pragma GCC unroll 1
    for (uint32_t fh = 0; fh < 2; ++fh) {
        const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_tile_addr) +
                                                 (2 * fh + col / FACE_HW) * FACE_ELEMS + (col % FACE_HW);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile_addr) + (2 * fh + half) * FACE_ELEMS;
        uint32_t bits = src[0];
#pragma GCC unroll 1
        for (uint32_t r = 1; r < FACE_HW; ++r) {
            const uint32_t nxt = src[r * FACE_HW];
#pragma GCC unroll 16
            for (uint32_t w = 0; w < FACE_HW; ++w) {
                dst[w] = bits;
            }
            dst += FACE_HW;
            bits = nxt;
        }
#pragma GCC unroll 16
        for (uint32_t w = 0; w < FACE_HW; ++w) {
            dst[w] = bits;
        }
    }
}

template <bool FAST>
FORCE_INLINE void expand_half(uint32_t raw_tile_addr, uint32_t col, uint32_t dst_tile_addr, uint32_t half) {
    if constexpr (FAST) {
        expand_half_fast(raw_tile_addr, col, dst_tile_addr, half);
    } else {
        expand_half_ref(raw_tile_addr, col, dst_tile_addr, half);
    }
}

// Expand global term g (stream j = g / (n+1), term t = g % (n+1)) of a set.
template <uint32_t n, uint32_t P, uint32_t coef_page_bytes, bool FAST>
FORCE_INLINE void expand_term(uint32_t g, uint32_t raw_post, uint32_t raw_comb, uint32_t bcast_base) {
    const uint32_t j = g / (n + 1);
    const uint32_t t = g - j * (n + 1);
    const uint32_t raw_addr = t == 0 ? raw_post : raw_comb;
    const uint32_t raw_col = t == 0 ? j : (t - 1) * n + j;
    const uint32_t tile = j * P + coef_tile_in_stream(t);
    expand_half<FAST>(raw_addr, raw_col, bcast_base + tile * coef_page_bytes, coef_half(t));
}

// Writer-side (BRISC) job. Without SHARE it expands every term; with SHARE the even-g terms and it pushes a
// stream only once the reader's share of it is signalled.
template <
    uint32_t n,
    uint32_t coef_tiles_per_stream,
    uint32_t post_tiles_per_row,
    uint32_t comb_tiles_per_row,
    uint32_t coef_page_bytes,
    uint32_t cb_coef_raw,
    uint32_t cb_coef_bcast,
    bool SHARE,
    bool FAST,
    class Accessor>
struct CoefExpander {
    static constexpr uint32_t P = coef_tiles_per_stream;
    static constexpr uint32_t num_coef_tiles = n * P;
    static constexpr uint32_t num_raw_tiles = post_tiles_per_row + comb_tiles_per_row;
    static constexpr uint32_t num_terms = n * (n + 1);
    static constexpr uint32_t stride = SHARE ? 2 : 1;

    const Accessor& post_acc;
    const Accessor& comb_acc;
    const uint32_t sem_go, sem_done;
    uint32_t raw_post_addr = 0, raw_comb_addr = 0, bcast_base = 0;
    uint32_t set_idx = 0;  // sets finished
    uint32_t g = 0;        // next own term
    uint32_t push_j = 0;   // next stream to push
    bool job_active = false;

    CoefExpander(const Accessor& post, const Accessor& comb, uint32_t go, uint32_t done) :
        post_acc(post), comb_acc(comb), sem_go(go), sem_done(done) {}

    bool active() const { return job_active; }
    bool own_done() const { return g >= num_terms; }

    void start(uint32_t row) {
        cb_reserve_back(cb_coef_raw, num_raw_tiles);
        raw_post_addr = get_write_ptr(cb_coef_raw);
        raw_comb_addr = raw_post_addr + post_tiles_per_row * coef_page_bytes;
        noc_async_read(post_acc.get_noc_addr(row * post_tiles_per_row), raw_post_addr, coef_page_bytes);
        noc_async_read(comb_acc.get_noc_addr(row * comb_tiles_per_row), raw_comb_addr, coef_page_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_coef_raw, num_raw_tiles);  // private scratch: push / wait / pop are bookkeeping
        cb_wait_front(cb_coef_raw, num_raw_tiles);
        cb_reserve_back(cb_coef_bcast, num_coef_tiles);
        bcast_base = get_write_ptr(cb_coef_bcast);
        g = 0;
        push_j = 0;
        job_active = true;
        if constexpr (SHARE) {
            cp_sem_write(sem_go, set_idx + 1);
        }
    }

    // Push every stream whose shares are all written; closes the job after the last.
    void try_push() {
        const uint32_t own_streams = own_done() ? n : g / (n + 1);
        while (push_j < own_streams) {
            if constexpr (SHARE) {
                if (cp_sem_read(sem_done) < set_idx * n + push_j + 1) {
                    return;
                }
            }
            cb_push_back(cb_coef_bcast, P);
            ++push_j;
        }
        if (push_j == n) {
            cb_pop_front(cb_coef_raw, num_raw_tiles);
            job_active = false;
            ++set_idx;
        }
    }

    // One own term (if any left), then the pushes it enables.
    void step() {
        if (!own_done()) {
            expand_term<n, P, coef_page_bytes, FAST>(g, raw_post_addr, raw_comb_addr, bcast_base);
            g += stride;
        }
        try_push();
    }
};

// Reader-side (NCRISC) share: the odd-g terms of every set, in order. Never touches a CB API of the coefficient
// CBs; addresses are derived from the fifo bases (see header).
template <uint32_t n, uint32_t P, uint32_t coef_page_bytes, uint32_t post_tiles_per_row, uint32_t coef_depth, bool FAST>
struct ReaderShare {
    static constexpr uint32_t num_terms = n * (n + 1);
    static constexpr uint32_t set_bytes = n * P * coef_page_bytes;
    const uint32_t sem_go, sem_done, raw_post, raw_comb, bcast0, total_sets;
    uint32_t set = 0, g = 1;
    bool active = false;

    ReaderShare(uint32_t go, uint32_t done, uint32_t raw_base, uint32_t bcast_base, uint32_t sets) :
        sem_go(go),
        sem_done(done),
        raw_post(raw_base),
        raw_comb(raw_base + post_tiles_per_row * coef_page_bytes),
        bcast0(bcast_base),
        total_sets(sets) {}

    bool finished() const { return set >= total_sets; }

    // Whether a term is ready to expand (opens the next set when the writer released it).
    bool ready() {
        if (set >= total_sets) {
            return false;
        }
        if (!active) {
            if (cp_sem_read(sem_go) < set + 1) {  // also invalidates the L1 cache: raw tiles are fresh
                return false;
            }
            active = true;
            g = 1;
        }
        return true;
    }

    // Expand one term (call only after ready()).
    void work() {
        const uint32_t base = bcast0 + (set % coef_depth) * set_bytes;
        expand_term<n, P, coef_page_bytes, FAST>(g, raw_post, raw_comb, base);
        const uint32_t j = g / (n + 1);
        g += 2;
        if (g >= num_terms) {
            cp_sem_write(sem_done, set * n + n);
            active = false;
            ++set;
        } else if (g / (n + 1) != j) {
            cp_sem_write(sem_done, set * n + j + 1);
        }
    }
};

}  // namespace mhc_post
