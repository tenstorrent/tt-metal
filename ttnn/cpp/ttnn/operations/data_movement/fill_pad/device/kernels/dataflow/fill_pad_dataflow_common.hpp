// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

static constexpr uint32_t FACE = 16;
static constexpr uint32_t TILE = 32;

/**
 * Write a mask tile into `p` using face layout.
 *
 * Positions where (gcol >= W_thresh OR grow >= H_thresh) receive `one_val`;
 * all other positions receive 0.
 * Pass TILE (32) for a threshold to disable that dimension's condition.
 */
template <typename T, uint32_t W_thresh, uint32_t H_thresh>
static void generate_mask_tile(T* __restrict__ p, T one_val) {
    T zero_val = static_cast<T>(0);
    for (uint32_t fr = 0; fr < 2; fr++) {
        for (uint32_t fc = 0; fc < 2; fc++) {
            const uint32_t face_base = (fr * 2u + fc) * FACE * FACE;
            for (uint32_t r = 0; r < FACE; r++) {
                const uint32_t grow = fr * FACE + r;
                for (uint32_t c = 0; c < FACE; c++) {
                    const uint32_t gcol = fc * FACE + c;
                    p[face_base + r * FACE + c] = (gcol >= W_thresh || grow >= H_thresh) ? one_val : zero_val;
                }
            }
        }
    }
}
