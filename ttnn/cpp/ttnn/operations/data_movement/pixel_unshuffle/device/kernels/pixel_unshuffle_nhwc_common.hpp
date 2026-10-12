// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_unshuffle, channels_last: the per-core work description and the L1 gather shared by the two
// dataflow RISCs (which also read the input rows from DRAM) and the compute RISCs (UNPACK and PACK,
// which only gather). Every RISC derives the same sequence of "items" (one image row of output
// pixels per dataflow RISC half) from the same compile-time geometry and runtime pixel range, so
// the readers never have to describe the work to the helpers; the circular buffer page counters
// are the only synchronisation.

#pragma once

#include <stdint.h>

namespace pixel_unshuffle_nhwc {

constexpr uint32_t cgcd(uint32_t a, uint32_t b) { return b == 0 ? a : cgcd(b, a % b); }

struct Item {
    uint32_t n;
    uint32_t ho;
    uint32_t va;     // first output column of this reader's span
    uint32_t vb;     // one past the last
    uint32_t ca_al;  // first input column read (aligned down)
    uint32_t cb_al;  // one past the last input column read (aligned up, <= W)
    uint32_t lpix0;  // shard-local index of pixel (n, ho, va)
};

template <
    uint32_t W,
    uint32_t C,
    uint32_t H,
    uint32_t r,
    uint32_t Ho,
    uint32_t Wo,
    uint32_t datum,
    uint32_t Cp,
    uint32_t out_row_nbytes,
    uint32_t aligned_row_nbytes_in,
    uint32_t channel_order>
struct Geom {
    static constexpr uint32_t SPATIAL_MAJOR = 1;
    static constexpr uint32_t HoWo = Ho * Wo;
    static constexpr uint32_t CH = C * H;
    static constexpr uint32_t C_out = C * r * r;
    static constexpr uint32_t rows_per_item = C * r;
    static constexpr uint32_t slot_nbytes = rows_per_item * aligned_row_nbytes_in;  // one CB page

    // DRAM reads want 32 B-aligned source offsets and matching L1 alignment, so a reader's input span is
    // aligned to ALIGN_ELEMS input columns. A split between the two readers at output column s puts the
    // input boundary at s*r, so s is a multiple of SPLIT_GRAN.
    static constexpr uint32_t ALIGN_ELEMS = 32 / datum;
    static constexpr uint32_t SPLIT_GRAN = ALIGN_ELEMS / cgcd(ALIGN_ELEMS, r);
    static constexpr bool fast = (datum == 2) && ((r % 2) == 0) && (channel_order != SPATIAL_MAJOR);

    // Walk the pixel range [pix0, end) one image row at a time and hand reader `risc` its half of the row.
    // Returns false when the range is exhausted. `cursor` is the global pixel index.
    static bool advance(uint32_t risc, uint32_t pix0, uint32_t end, uint32_t& cursor, Item& it) {
        while (cursor < end) {
            const uint32_t n = cursor / HoWo;
            const uint32_t rem = cursor - n * HoWo;
            const uint32_t ho = rem / Wo;
            const uint32_t wa = rem - ho * Wo;
            const uint32_t row_len = (end - cursor < Wo - wa) ? (end - cursor) : (Wo - wa);
            const uint32_t wb = wa + row_len;
            const uint32_t row_first = cursor;
            cursor += row_len;

            const uint32_t half = ((row_len / 2) / SPLIT_GRAN) * SPLIT_GRAN;
            uint32_t va;
            uint32_t vb;
            if (half == 0) {  // too short to split: reader 0 takes it all
                if (risc != 0) {
                    continue;
                }
                va = wa;
                vb = wb;
            } else if (risc == 0) {
                va = wa;
                vb = wa + half;
            } else {
                va = wa + half;
                vb = wb;
            }
            it.n = n;
            it.ho = ho;
            it.va = va;
            it.vb = vb;
            it.ca_al = (va * r) & ~(ALIGN_ELEMS - 1);
            const uint32_t cb = vb * r;
            const uint32_t cb_up = (cb + ALIGN_ELEMS - 1) & ~(ALIGN_ELEMS - 1);
            it.cb_al = cb_up < W ? cb_up : W;
            it.lpix0 = row_first - pix0 + (va - wa);
            return true;
        }
        return false;
    }

    // The sub-span [a, b) of item `it` that covers 24ths [lo, hi) of its pixels.
    static void part(const Item& it, uint32_t lo, uint32_t hi, uint32_t& a, uint32_t& b) {
        const uint32_t len = it.vb - it.va;
        a = it.va + (len * lo) / 24;
        b = it.va + (len * hi) / 24;
    }

    // Gather output pixels [va, vb) of `it` from the staged input rows at `slot_base` into the shard at
    // `out_base`, with plain 32-bit L1 loads and stores. Fast path (2-byte elements, r even,
    // CHANNEL_MAJOR): pixel v takes, from input row (c, dy), the r elements at columns [v*r, v*r + r) -
    // r/2 naturally aligned words - and puts them at channels [c*r*r + dy*r, +r): also r/2 aligned words.
    static void gather(const Item& it, uint32_t va, uint32_t vb, uint32_t slot_base, uint32_t out_base) {
        if (va >= vb) {
            return;
        }
        uint32_t dst = out_base + (it.lpix0 + (va - it.va)) * out_row_nbytes;

        if constexpr (fast) {
            constexpr uint32_t wpr = r / 2;      // words per (c, dy) per pixel
            constexpr uint32_t wpp = C_out / 2;  // data words per pixel
            constexpr uint32_t wpad = Cp / 2;    // total words per pixel
            const uint32_t* rows[rows_per_item];
            for (uint32_t k = 0; k < rows_per_item; k++) {
                rows[k] = (const uint32_t*)(slot_base + k * aligned_row_nbytes_in);
            }
            uint32_t v = va;
            uint32_t sw = (va * r - it.ca_al) / 2;  // source word index of pixel v
            if constexpr (2 * wpp <= 16) {
                // Small pixels (r=2, C<=4): two pixels per iteration, all 2*wpp loads issued before any store.
                for (; v + 1 < vb; v += 2) {
                    uint32_t w0[wpp];
                    uint32_t w1[wpp];
#pragma GCC unroll 32
                    for (uint32_t k = 0; k < rows_per_item; k++) {
#pragma GCC unroll 8
                        for (uint32_t t = 0; t < wpr; t++) {
                            w0[k * wpr + t] = rows[k][sw + t];
                            w1[k * wpr + t] = rows[k][sw + wpr + t];
                        }
                    }
                    tt_l1_ptr uint32_t* d0 = (tt_l1_ptr uint32_t*)dst;
                    tt_l1_ptr uint32_t* d1 = (tt_l1_ptr uint32_t*)(dst + out_row_nbytes);
#pragma GCC unroll 32
                    for (uint32_t q = 0; q < wpp; q++) {
                        d0[q] = w0[q];
                        d1[q] = w1[q];
                    }
#pragma GCC unroll 32
                    for (uint32_t q = wpp; q < wpad; q++) {
                        d0[q] = 0;
                        d1[q] = 0;
                    }
                    dst += 2 * out_row_nbytes;
                    sw += 2 * wpr;
                }
            } else {
                // Large pixels (e.g. r=4, C=3: 24 words): one pixel per iteration, gathered in chunks of CHUNK
                // words so about 12 loads are in flight without spilling registers.
                constexpr uint32_t CHUNK = 12;
                for (; v < vb; v++) {
                    tt_l1_ptr uint32_t* d0 = (tt_l1_ptr uint32_t*)dst;
#pragma GCC unroll 8
                    for (uint32_t q0 = 0; q0 < wpp; q0 += CHUNK) {
                        constexpr uint32_t n = CHUNK;
                        uint32_t w[n];
#pragma GCC unroll 16
                        for (uint32_t j = 0; j < n; j++) {
                            if (q0 + j < wpp) {
                                const uint32_t k = (q0 + j) / wpr;
                                w[j] = rows[k][sw + ((q0 + j) % wpr)];
                            }
                        }
#pragma GCC unroll 16
                        for (uint32_t j = 0; j < n; j++) {
                            if (q0 + j < wpp) {
                                d0[q0 + j] = w[j];
                            }
                        }
                    }
#pragma GCC unroll 32
                    for (uint32_t q = wpp; q < wpad; q++) {
                        d0[q] = 0;
                    }
                    dst += out_row_nbytes;
                    sw += wpr;
                }
            }
            if (v < vb) {  // odd tail of the pair loop
                tt_l1_ptr uint32_t* d0 = (tt_l1_ptr uint32_t*)dst;
#pragma GCC unroll 32
                for (uint32_t k = 0; k < rows_per_item; k++) {
#pragma GCC unroll 8
                    for (uint32_t t = 0; t < wpr; t++) {
                        d0[k * wpr + t] = rows[k][sw + t];
                    }
                }
#pragma GCC unroll 32
                for (uint32_t q = wpp; q < wpad; q++) {
                    d0[q] = 0;
                }
            }
        } else if constexpr (datum == 2) {
            // Element path, 16-bit: c_out(c, dy, dx) per channel_order.
            const uint16_t* rows[rows_per_item];
            for (uint32_t k = 0; k < rows_per_item; k++) {
                rows[k] = (const uint16_t*)(slot_base + k * aligned_row_nbytes_in);
            }
            for (uint32_t v = va; v < vb; v++) {
                tt_l1_ptr uint16_t* d = (tt_l1_ptr uint16_t*)dst;
                const uint32_t se = v * r - it.ca_al;
                for (uint32_t c = 0; c < C; c++) {
                    for (uint32_t dy = 0; dy < r; dy++) {
                        const uint16_t* srow = rows[c * r + dy] + se;
                        for (uint32_t dx = 0; dx < r; dx++) {
                            const uint32_t co = (channel_order == SPATIAL_MAJOR) ? (dy * (r * C) + dx * C + c)
                                                                                 : (c * r * r + dy * r + dx);
                            d[co] = srow[dx];
                        }
                    }
                }
                for (uint32_t co = C_out; co < Cp; co++) {
                    d[co] = 0;
                }
                dst += out_row_nbytes;
            }
        } else {
            // Element path, 32-bit elements.
            const uint32_t* rows[rows_per_item];
            for (uint32_t k = 0; k < rows_per_item; k++) {
                rows[k] = (const uint32_t*)(slot_base + k * aligned_row_nbytes_in);
            }
            for (uint32_t v = va; v < vb; v++) {
                tt_l1_ptr uint32_t* d = (tt_l1_ptr uint32_t*)dst;
                const uint32_t se = v * r - it.ca_al;
                for (uint32_t c = 0; c < C; c++) {
                    for (uint32_t dy = 0; dy < r; dy++) {
                        const uint32_t* srow = rows[c * r + dy] + se;
                        for (uint32_t dx = 0; dx < r; dx++) {
                            const uint32_t co = (channel_order == SPATIAL_MAJOR) ? (dy * (r * C) + dx * C + c)
                                                                                 : (c * r * r + dy * r + dx);
                            d[co] = srow[dx];
                        }
                    }
                }
                for (uint32_t co = C_out; co < Cp; co++) {
                    d[co] = 0;
                }
                dst += out_row_nbytes;
            }
        }
    }
};

}  // namespace pixel_unshuffle_nhwc
