// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// x multicast family of this kernel (host define Q_STEM_X_FAMILY: the rotating injector index, or the K group under
// K-split): named args "mcast<F>_ct_offset" / "mcast<F>_rt_offset"
#ifndef Q_STEM_X_FAMILY
#define Q_STEM_X_FAMILY 0
#endif
#define Q_STEM_STR2(x) #x
#define Q_STEM_STR(x) Q_STEM_STR2(x)
#define Q_STEM_X_FAMILY_ARG(suffix) "mcast" Q_STEM_STR(Q_STEM_X_FAMILY) suffix

namespace q_stem {

// The stage-1 send order — the single device-side definition shared by the injector, the stage-1 writers and the
// ports (host twin: q_stem_program_descriptor.remote_order + Geometry.num_blocks).
//
//   remote blocks first, interleaved across destinations (every direction busy from the first block):
//     for b < B: for r0 in 0, m, 2m, .. < qrt: for i < G-1: (b, remote_order[i], r0)
//   then the own quarter (no fabric; feeds the local heads):
//     for b < B: for r0: (b, p, r0)
//
// A block is m row tiles (rows r0 .. r0+m-1 of quarter d of image b); block j holds q rows j*m .. j*m+m-1 of the
// core's cb_q_slice. f(j, b, d, r0).
template <typename F>
inline void for_each_block(
    uint32_t B, uint32_t qrt, uint32_t m, uint32_t G, uint32_t p, const uint32_t* remote_order, F&& f) {
    uint32_t j = 0;
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t r0 = 0; r0 < qrt; r0 += m) {
            for (uint32_t i = 0; i + 1 < G; ++i) {
                f(j++, b, remote_order[i], r0);
            }
        }
    }
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t r0 = 0; r0 < qrt; r0 += m) {
            f(j++, b, p, r0);
        }
    }
}

// Stage-3 consumption order of a core's rows = its landing layout (host twin: landing_slot / Geometry.ts).
// Row c (0 .. hpc * rows - 1) -> (head u in consumption order, row t). Remote heads u < n_r, local heads after:
//   remote rows t < ts, row-major across the remote heads (they land with the same row profile)
//   then the local heads, head-major (the own quarter lands when stage 1 ends)
//   then the late remote rows t >= ts, row-major (they land last, behind the busiest fabric link)
// With no local head (or no remote head) the split vanishes: ts = rows.
struct RowOrder {
    uint32_t n_r, hpc, rows, ts;
    RowOrder(uint32_t n_r_, uint32_t hpc_, uint32_t rows_, uint32_t ts_) :
        n_r(n_r_), hpc(hpc_), rows(rows_), ts(n_r_ < hpc_ ? ts_ : rows_) {}
    uint32_t total() const { return hpc * rows; }
    uint32_t first_local() const { return n_r * ts; }
    void at(uint32_t c, uint32_t& u, uint32_t& t) const {
        const uint32_t a = n_r * ts;
        const uint32_t l = (hpc - n_r) * rows;
        if (c < a) {
            t = c / n_r;
            u = c - t * n_r;
        } else if (c < a + l) {
            const uint32_t k = c - a;
            const uint32_t lh = k / rows;
            u = n_r + lh;
            t = k - lh * rows;
        } else {
            const uint32_t k = c - a - l;
            const uint32_t tt = k / n_r;
            u = k - tt * n_r;
            t = ts + tt;
        }
    }
};

inline uint32_t unpack_x(uint32_t v) { return v >> 16; }
inline uint32_t unpack_y(uint32_t v) { return v & 0xFFFF; }

}  // namespace q_stem
