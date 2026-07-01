// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// batch_fft_common.h — Shared layout for the device-side BATCH FFT
// (Optimisation 1 of fft_stockham).

#pragma once

constexpr uint32_t CB_EVEN_R    = 0;
constexpr uint32_t CB_EVEN_I    = 1;
constexpr uint32_t CB_ODD_R     = 2;
constexpr uint32_t CB_ODD_I     = 3;
constexpr uint32_t CB_TW_R      = 4;
constexpr uint32_t CB_TW_I      = 5;
constexpr uint32_t CB_OUT0_R    = 6;
constexpr uint32_t CB_OUT0_I    = 7;
constexpr uint32_t CB_OUT1_R    = 8;
constexpr uint32_t CB_OUT1_I    = 9;
constexpr uint32_t CB_TMP_R     = 10;
constexpr uint32_t CB_TMP_I     = 11;
constexpr uint32_t CB_TW_ODD_R  = 12;
constexpr uint32_t CB_TW_ODD_I  = 13;
constexpr uint32_t CB_STATE_R   = 14;
constexpr uint32_t CB_STATE_I   = 15;
constexpr uint32_t CB_SYNC      = 16;

// ── BF16 I/O staging (commit 2 of host-to-device refactor) ────────────
// Only allocated by SingleTileStockhamFactory when input dtype is BFLOAT16.
// Legacy stockham_host path leaves these unallocated and passes
// INPUT_BF16=0 / OUTPUT_BF16=0 so the reader/writer never reference them.
constexpr uint32_t CB_IN_R_BF16  = 17;
constexpr uint32_t CB_IN_I_BF16  = 18;
constexpr uint32_t CB_OUT_R_BF16 = 19;
constexpr uint32_t CB_OUT_I_BF16 = 20;

constexpr uint32_t BATCH_NUM_CBS      = 17;   // fp32-only path
constexpr uint32_t BATCH_NUM_CBS_BF16 = 21;   // fp32 internal + bf16 I/O

constexpr uint32_t TILE_HW        = 32;
constexpr uint32_t TILE_ELEMS     = TILE_HW * TILE_HW;       // 1024
constexpr uint32_t TILE_SIZE_FP32 = TILE_ELEMS * 4;          // 4096 bytes
constexpr uint32_t TILE_SIZE_BF16 = TILE_ELEMS * 2;          // 2048 bytes
