// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * Write a mask tile into `p` using face layout.
 *
 * Positions where (gcol >= W_thresh OR grow >= H_thresh) receive `one_val`;
 * all other positions receive 0. Pass TILE for a threshold to disable that
 * dimension's condition. TILE is the edge length of the tile in elements;
 * a 2x2 face layout is assumed (FACE = TILE / 2).
 */
template <typename T, uint32_t W_thresh, uint32_t H_thresh, uint32_t TILE>
void generate_mask_tile(T* __restrict__ p, T one_val) {
    constexpr uint32_t FACE = TILE / 2u;
    // A face row is zeros up to column c0 and ones from there on, so rows that are all zeros or all ones
    // are written as 32-bit words; only a row with c0 inside it is written element by element.
    constexpr uint32_t row_words = FACE * sizeof(T) / 4u;
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "mask element must be 16 or 32 bit");
    const uint32_t one_word =
        (sizeof(T) == 2) ? ((static_cast<uint32_t>(one_val) & 0xFFFFu) * 0x10001u) : static_cast<uint32_t>(one_val);
    for (uint32_t fr = 0; fr < 2; fr++) {
        for (uint32_t fc = 0; fc < 2; fc++) {
            T* face = p + (fr * 2u + fc) * FACE * FACE;
            const uint32_t col0 = fc * FACE;
            const uint32_t w_start = (col0 >= W_thresh) ? 0u : ((W_thresh >= col0 + FACE) ? FACE : (W_thresh - col0));
            for (uint32_t r = 0; r < FACE; r++) {
                volatile uint32_t* row = reinterpret_cast<volatile uint32_t*>(face + r * FACE);
                const uint32_t c0 = (fr * FACE + r >= H_thresh) ? 0u : w_start;
                if (c0 == FACE) {
#pragma GCC unroll 16
                    for (uint32_t k = 0; k < row_words; k++) {
                        row[k] = 0u;
                    }
                } else if (c0 == 0u) {
#pragma GCC unroll 16
                    for (uint32_t k = 0; k < row_words; k++) {
                        row[k] = one_word;
                    }
                } else {
                    T* e = face + r * FACE;
#pragma GCC unroll 16
                    for (uint32_t c = 0; c < FACE; c++) {
                        e[c] = (c >= c0) ? one_val : static_cast<T>(0);
                    }
                }
            }
        }
    }
}

/**
 * Reserve a single tile in `cb_right_mask`, write a right-edge mask
 * (1 at gcol >= W_mod32, 0 elsewhere) and push it. Used by both the unified
 * and sharded writer kernels during Phase 1.
 */
template <typename MASK_T, uint32_t W_mod32, uint32_t TILE, typename CB_T>
void push_right_mask_tile(CB_T& cb_right_mask, MASK_T one_val) {
    cb_right_mask.reserve_back(1);
    generate_mask_tile<MASK_T, W_mod32, TILE, TILE>(reinterpret_cast<MASK_T*>(cb_right_mask.get_write_ptr()), one_val);
    cb_right_mask.push_back(1);
}

/**
 * Reserve a single tile in `cb_bot_mask`, write a bottom-edge mask
 * (1 at grow >= H_mod32, 0 elsewhere) and push it.
 */
template <typename MASK_T, uint32_t H_mod32, uint32_t TILE, typename CB_T>
void push_bottom_mask_tile(CB_T& cb_bot_mask, MASK_T one_val) {
    cb_bot_mask.reserve_back(1);
    generate_mask_tile<MASK_T, TILE, H_mod32, TILE>(reinterpret_cast<MASK_T*>(cb_bot_mask.get_write_ptr()), one_val);
    cb_bot_mask.push_back(1);
}
