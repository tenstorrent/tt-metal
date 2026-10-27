// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_common.h — shared tile geometry for the device-side
// apply_twiddles op (the between-pass elementwise complex multiply of
// Cooley–Tukey two-pass FFT) and the complex_mul / apply_twiddles_xl ops
// that reuse its writer and compute kernels.
//
// The dataflow buffers themselves are named resources (dfb::a_r, dfb::t_r,
// dfb::b_r, dfb::tmp_r, and the bf16 staging pair) declared by each
// factory's ProgramSpec, so this header no longer hard-codes CB indices.
// The bf16 staging buffers exist only when the INPUT_BF16 / OUTPUT_BF16
// defines are set, matching the factories' conditional allocation.
//
// Tile layout (kTileElems = 1024 fp32 elements per tile = 4096 B):
//   - Input/output rows of length N1 occupy slots [0, N1) of each tile.
//     Slots [N1, kTileElems) may contain garbage in compute; the writer
//     only emits N1*elem_size bytes per row so garbage never reaches DRAM.
//   - Twiddle rows are tile-padded on the host: slots [N1, kTileElems) = 0.

#pragma once

#include <cstdint>

constexpr uint32_t kTileHW = 32u;
constexpr uint32_t kTileElems = kTileHW * kTileHW;    // 1024
constexpr uint32_t kTileBytesFp32 = kTileElems * 4u;  // 4096
constexpr uint32_t kTileBytesBf16 = kTileElems * 2u;  // 2048
