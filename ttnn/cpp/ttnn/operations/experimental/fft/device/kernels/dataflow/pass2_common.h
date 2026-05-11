// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pass2_common.h — Shared CB layout for the device-side Pass-2 kernel
// (twiddle multiply step of Stockham). Tile-granular, full-IEEE-fp32.

#pragma once

constexpr uint32_t CB_A_R   = 0;   // input row tile, real
constexpr uint32_t CB_A_I   = 1;   // input row tile, imag
constexpr uint32_t CB_T_R   = 2;   // twiddle row tile, real
constexpr uint32_t CB_T_I   = 3;   // twiddle row tile, imag
constexpr uint32_t CB_B_R   = 4;   // output (A*T) row tile, real
constexpr uint32_t CB_B_I   = 5;   // output (A*T) row tile, imag
constexpr uint32_t CB_TMP_R = 6;   // cmul scratch
constexpr uint32_t CB_TMP_I = 7;

constexpr uint32_t PASS2_NUM_CBS = 8;

constexpr uint32_t TILE_HW        = 32;
constexpr uint32_t TILE_ELEMS     = TILE_HW * TILE_HW;       // 1024
constexpr uint32_t TILE_SIZE_FP32 = TILE_ELEMS * 4;          // 4096 bytes
