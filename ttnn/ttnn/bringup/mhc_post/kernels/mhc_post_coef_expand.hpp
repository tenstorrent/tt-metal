// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post — load_coefficients (dataflow only; run by the writer-role DM kernel (BRISC) of mhc_post_dm.cpp, the
// single producer of cb_coef_raw / cb_coef_bcast).
//
// Per segment (one token-tile row r): read the raw post / comb tiles of row r into cb_coef_raw (private
// scratch of the expanding kernel), then expand them into n * P half-packed column-broadcast fp32 tiles in
// cb_coef_bcast (layout: mhc_post_common.hpp — term t of stream j in half t%2 of tile j*P + t/2; term 0 =
// post(rho, j), term 1+i = comb(rho, i*n + j)), pushing each output stream's P tiles as soon as they are
// written: compute's stream-j wait needs only streams <= j, so it mixes stream j while later streams are
// still being expanded. Values are copied bit-for-bit (fp32 words).
//
// The load is a RESUMABLE JOB (start() then one step() per half-tile term), so the event loop of the kernel that
// runs it can interleave the expansion with its other duties (output writes, read help) and a pending request
// waits at most one term, not a whole set.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_post_common.hpp"

namespace mhc_post {

constexpr uint32_t FACE_HW = 16;
constexpr uint32_t FACE_ELEMS = FACE_HW * FACE_HW;

// Half-tile column-broadcast expansion: every element (rho, gamma) of half `half` (gamma in
// [16*half, 16*half + 16)) of the fp32 tile at dst_tile_addr becomes raw(rho, col) (col < 32).
// The half is two contiguous faces (2*fh + half, fh = row half), each 16 rows x 16 words; one raw load per
// row feeds 16 unrolled word stores (no per-element address math, no per-row helper call). Software-pipelined
// (Perf 2, coef_prefetch): the load of face row r+1 is issued before the 16 stores of row r, so the stores do not
// stall on every row's load-use (measured 13.3 -> 11.6 us per n = 4 set on Blackhole).
FORCE_INLINE void expand_half(uint32_t raw_tile_addr, uint32_t col, uint32_t dst_tile_addr, uint32_t half) {
#pragma GCC unroll 1
    for (uint32_t fh = 0; fh < 2; ++fh) {
        const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw_tile_addr) +
                                                 (2 * fh + col / FACE_HW) * FACE_ELEMS + (col % FACE_HW);
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile_addr) + (2 * fh + half) * FACE_ELEMS;
        uint32_t bits = src[0];
#pragma GCC unroll 1
        for (uint32_t r = 1; r < FACE_HW; ++r) {
            const uint32_t next_bits = src[r * FACE_HW];
#pragma GCC unroll 16
            for (uint32_t w = 0; w < FACE_HW; ++w) {
                dst[w] = bits;
            }
            dst += FACE_HW;
            bits = next_bits;
        }
#pragma GCC unroll 16
        for (uint32_t w = 0; w < FACE_HW; ++w) {
            dst[w] = bits;
        }
    }
}

template <
    uint32_t n,
    uint32_t coef_tiles_per_stream,
    uint32_t post_tiles_per_row,
    uint32_t comb_tiles_per_row,
    uint32_t coef_page_bytes,
    uint32_t cb_coef_raw,
    uint32_t cb_coef_bcast,
    class Accessor>
struct CoefExpander {
    static constexpr uint32_t num_coef_tiles = n * coef_tiles_per_stream;
    static constexpr uint32_t num_raw_tiles = post_tiles_per_row + comb_tiles_per_row;

    const Accessor& post_acc;
    const Accessor& comb_acc;
    uint32_t raw_post_addr = 0;
    uint32_t raw_comb_addr = 0;

    CoefExpander(const Accessor& post, const Accessor& comb) : post_acc(post), comb_acc(comb) {}

    uint32_t bcast_base = 0;
    uint32_t job_j = 0, job_t = 0;
    bool job_active = false;

    bool active() const { return job_active; }

    // Reserve cb_coef_raw and issue the raw post / comb reads of token row `row` (no barrier).
    void issue_raw_reads(uint32_t row) {
        cb_reserve_back(cb_coef_raw, num_raw_tiles);
        raw_post_addr = get_write_ptr(cb_coef_raw);
        raw_comb_addr = raw_post_addr + post_tiles_per_row * coef_page_bytes;
        noc_async_read(post_acc.get_noc_addr(row * post_tiles_per_row), raw_post_addr, coef_page_bytes);
        noc_async_read(comb_acc.get_noc_addr(row * comb_tiles_per_row), raw_comb_addr, coef_page_bytes);
    }

    // Open the job for token row `row`: raw reads + their own barrier, then the cb_coef_bcast window.
    void start(uint32_t row) {
        issue_raw_reads(row);
        noc_async_read_barrier();
        cb_push_back(cb_coef_raw, num_raw_tiles);  // private scratch: push / wait / pop are bookkeeping
        cb_wait_front(cb_coef_raw, num_raw_tiles);
        cb_reserve_back(cb_coef_bcast, num_coef_tiles);
        bcast_base = get_write_ptr(cb_coef_bcast);
        job_j = job_t = 0;
        job_active = true;
    }

    // Expand one term (half-tile) of the open job; pushes stream j's P tiles after its last term.
    void step() {
        const uint32_t t = job_t;
        const uint32_t raw_addr = t == 0 ? raw_post_addr : raw_comb_addr;
        const uint32_t raw_col = t == 0 ? job_j : (t - 1) * n + job_j;
        const uint32_t tile = job_j * coef_tiles_per_stream + coef_tile_in_stream(t);
        expand_half(raw_addr, raw_col, bcast_base + tile * coef_page_bytes, coef_half(t));
        if (++job_t > n) {
            cb_push_back(cb_coef_bcast, coef_tiles_per_stream);
            job_t = 0;
            if (++job_j == n) {
                cb_pop_front(cb_coef_raw, num_raw_tiles);
                job_active = false;
            }
        }
    }
};

}  // namespace mhc_post
