// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ── Circular Buffer indices ────────────────────────────────────────────────
constexpr uint32_t CB_EVEN_R    = 0;   // lo element of each butterfly pair
constexpr uint32_t CB_EVEN_I    = 1;
constexpr uint32_t CB_ODD_R     = 2;   // hi element of each butterfly pair
constexpr uint32_t CB_ODD_I     = 3;
constexpr uint32_t CB_TW_R      = 4;   // stage twiddle factors (per pair)
constexpr uint32_t CB_TW_I      = 5;
constexpr uint32_t CB_OUT0_R    = 6;   // even + W*odd
constexpr uint32_t CB_OUT0_I    = 7;
constexpr uint32_t CB_OUT1_R    = 8;   // even - W*odd
constexpr uint32_t CB_OUT1_I    = 9;
constexpr uint32_t CB_TMP_R     = 10;  // cmul intermediates
constexpr uint32_t CB_TMP_I     = 11;
constexpr uint32_t CB_TW_ODD_R  = 12;  // W * odd
constexpr uint32_t CB_TW_ODD_I  = 13;
constexpr uint32_t CB_STATE_R   = 14;  // persistent state (reader-owned)
constexpr uint32_t CB_STATE_I   = 15;
constexpr uint32_t CB_SYNC      = 16;  // reader -> writer signal
constexpr uint32_t CB_RECV_R    = 17;  // receive buffer for partner's state
constexpr uint32_t CB_RECV_I    = 18;  //   (cross-core stages only)

constexpr uint32_t NUM_CBS = 19;

// ── Tile geometry ──────────────────────────────────────────────────────────
constexpr uint32_t TILE_HW        = 32;
constexpr uint32_t TILE_ELEMS     = TILE_HW * TILE_HW;       // 1024
constexpr uint32_t TILE_SIZE_FP32 = TILE_ELEMS * 4;          // 4096 bytes
