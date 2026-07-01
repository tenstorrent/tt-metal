// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_common.h — shared CB layout for the device-side
// apply_twiddles op (the between-pass elementwise complex multiply of
// Cooley–Tukey two-pass FFT).
//
// CB IDs 0..7 intentionally mirror pass2_common.h so apply_twiddles_compute
// is bit-identical to pass2_compute (same SFPU complex-multiply pipeline).
// CB IDs 8..11 are added for the bf16 staging tiles used when the input /
// output tensors are bf16 (in-kernel expand/truncate at the DRAM boundary;
// internal compute stays fp32).
//
// Tile layout (kTileElems = 1024 fp32 elements per tile = 4096 B):
//   - Input/output rows of length N1 occupy slots [0, N1) of each tile.
//     Slots [N1, kTileElems) may contain garbage in compute; the writer
//     only emits N1*elem_size bytes per row so garbage never reaches DRAM.
//   - Twiddle rows are tile-padded on the host: slots [N1, kTileElems) = 0.

#pragma once

#include <cstdint>

// ── fp32 compute CBs (must match pass2_common.h) ───────────────────────
constexpr uint32_t CB_A_R   = 0;   // input row tile, real (fp32)
constexpr uint32_t CB_A_I   = 1;   // input row tile, imag (fp32)
constexpr uint32_t CB_T_R   = 2;   // twiddle row tile, real (fp32)
constexpr uint32_t CB_T_I   = 3;   // twiddle row tile, imag (fp32)
constexpr uint32_t CB_B_R   = 4;   // output row tile, real (fp32)
constexpr uint32_t CB_B_I   = 5;   // output row tile, imag (fp32)
constexpr uint32_t CB_TMP_R = 6;   // SFPU cmul scratch
constexpr uint32_t CB_TMP_I = 7;

constexpr uint32_t APPLY_TW_NUM_FP32_CBS = 8;

// ── bf16 staging CBs (only allocated when INPUT_BF16 / OUTPUT_BF16) ────
// Reader reads bf16 tiles into IN_R_BF16/IN_I_BF16 then expands them to
// fp32 in CB_A_R/CB_A_I. Writer truncates CB_B_R/CB_B_I to bf16 in
// OUT_R_BF16/OUT_I_BF16 and DMAs those out.
constexpr uint32_t CB_IN_R_BF16  = 8;
constexpr uint32_t CB_IN_I_BF16  = 9;
constexpr uint32_t CB_OUT_R_BF16 = 10;
constexpr uint32_t CB_OUT_I_BF16 = 11;

constexpr uint32_t kTileHW    = 32u;
constexpr uint32_t kTileElems = kTileHW * kTileHW;          // 1024
constexpr uint32_t kTileBytesFp32 = kTileElems * 4u;        // 4096
constexpr uint32_t kTileBytesBf16 = kTileElems * 2u;        // 2048
