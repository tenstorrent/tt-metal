// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_run_fill bench: the slab bands of one M-block (conv2d_3x3 interleaved walk, stride 1, d 1, W_out == W) and the
// chunk walk shared by reader, compute and writer. Mirrored on the host by bench_brf.bands().
#pragma once

#include <cstdint>

namespace brf {

struct Bands {
    int32_t lo[3];
    int32_t hi[3];
};

template <
    int32_t W,
    int32_t num_in_px,
    int32_t num_out_px,
    uint32_t bm_,
    int32_t pad_top,
    int32_t pad_left,
    uint32_t unused = 0>
struct Geom {
    static constexpr uint32_t bm = bm_;
    static Bands bands(uint32_t m_tile_start, uint32_t actual_m_tiles, uint32_t m) {
        const uint32_t done = m * bm;
        const uint32_t tiles = actual_m_tiles - done < bm ? actual_m_tiles - done : bm;
        const int32_t o_first = static_cast<int32_t>((m_tile_start + done) * 32);
        int32_t o_end = static_cast<int32_t>((m_tile_start + done + tiles) * 32);
        o_end = o_end < num_out_px ? o_end : num_out_px;
        Bands b;
        for (int32_t kh = 0; kh < 3; ++kh) {
            const int32_t shift = (kh - pad_top) * W - pad_left;
            int32_t lo = o_first + shift;
            int32_t hi = o_end + shift + 2;
            lo = lo > 0 ? lo : 0;
            hi = hi < num_in_px ? hi : num_in_px;
            b.lo[kh] = lo;
            b.hi[kh] = hi > lo ? hi : lo;
        }
        return b;
    }
};

// Blocks x bands x chunks [a, e) of CH pixels, in order.
template <typename G, uint32_t CH>
struct ChunkCursor {
    uint32_t m_tile_start = 0, actual = 0, num_blocks = 0;
    uint32_t m = 0, kh = 0;
    int32_t a = 0, e = 0, band_lo = 0;
    bool done = false;
    Bands b{};

    void start(uint32_t start_tile, uint32_t actual_tiles) {
        m_tile_start = start_tile;
        actual = actual_tiles;
        num_blocks = (actual + G::bm - 1) / G::bm;
        m = 0;
        kh = 0;
        done = num_blocks == 0;
        if (!done) {
            b = G::bands(m_tile_start, actual, 0);
            a = b.lo[0];
            settle();
        }
    }
    // Move to the first non-empty band at or after (m, kh) with a = its start when a was reset.
    void settle() {
        while (!done && a >= b.hi[kh]) {
            if (++kh == 3) {
                kh = 0;
                if (++m == num_blocks) {
                    done = true;
                    return;
                }
                b = G::bands(m_tile_start, actual, m);
            }
            a = b.lo[kh];
        }
        band_lo = b.lo[kh];
        e = a + static_cast<int32_t>(CH) < b.hi[kh] ? a + static_cast<int32_t>(CH) : b.hi[kh];
    }
    void next() {
        a += CH;
        settle();
    }
};

}  // namespace brf
