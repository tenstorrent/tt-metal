// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ─────────────────────────────────────────────────────────────────────────
//  DUPLICATE — keep byte-for-byte in sync with

#pragma once

constexpr uint32_t CB_A       = 0;  // streaming A input  (in_R or in_I, per round)
constexpr uint32_t CB_B       = 1;  // streaming B input  (T_R / T_I / T_I_neg, per round)
constexpr uint32_t CB_OUT_R   = 2;  // output tile, real part
constexpr uint32_t CB_OUT_I   = 3;  // output tile, imag part

constexpr uint32_t PACKED_DFT_BF16_NUM_CBS = 4;

constexpr uint32_t PACKED_ROWS_PER_TILE = 32;   // sub-FFTs per tile
constexpr uint32_t TILE_HW              = 32;
constexpr uint32_t TILE_ELEMS           = TILE_HW * TILE_HW;   // 1024
constexpr uint32_t TILE_SIZE_BF16       = TILE_ELEMS * 2;       // 2048 bytes
