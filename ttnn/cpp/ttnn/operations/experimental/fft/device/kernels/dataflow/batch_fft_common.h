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

constexpr uint32_t BATCH_NUM_CBS = 17;   // no RECV CBs in batch mode

constexpr uint32_t TILE_HW        = 32;
constexpr uint32_t TILE_ELEMS     = TILE_HW * TILE_HW;       // 1024
constexpr uint32_t TILE_SIZE_FP32 = TILE_ELEMS * 4;          // 4096 bytes
