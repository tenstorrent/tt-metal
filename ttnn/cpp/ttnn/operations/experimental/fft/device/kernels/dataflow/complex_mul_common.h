// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// complex_mul_common.h — Shared CB layout for the elementwise complex-multiply
// kernel used by the Bluestein chirp pre/post multiplies in
// `device/universal_host.hpp` (behind the TT_FFT_DEVICE_CHIRP_MUL toggle).
//
// Math performed on tile by complex_mul_compute.cpp:
//     out[i] = a[i] * b[i]
//     re_out = re_a * re_b - im_a * im_b
//     im_out = re_a * im_b + im_a * re_b
//
// `a` advances one tile per step. `b` advances one tile per step but wraps
// modulo `num_b_tiles` so the same chirp can be broadcast across many input
// rows without re-uploading it.

#pragma once

constexpr uint32_t CB_A_R    = 0;   // input  signal, real
constexpr uint32_t CB_A_I    = 1;   // input  signal, imag
constexpr uint32_t CB_B_R    = 2;   // chirp / second operand, real
constexpr uint32_t CB_B_I    = 3;   // chirp / second operand, imag
constexpr uint32_t CB_OUT_R  = 4;   // output (a*b), real
constexpr uint32_t CB_OUT_I  = 5;   // output (a*b), imag
constexpr uint32_t CB_TMP_R  = 6;   // SFPU cmul scratch
constexpr uint32_t CB_TMP_I  = 7;

constexpr uint32_t CMUL_NUM_CBS = 8;

constexpr uint32_t TILE_HW        = 32;
constexpr uint32_t TILE_ELEMS     = TILE_HW * TILE_HW;     // 1024
constexpr uint32_t TILE_SIZE_FP32 = TILE_ELEMS * 4;        // 4096 bytes
