// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// batch_fft_common.h — Shared element/page geometry for device-side BATCH FFT.
// Metal 2.0 assigns DFB indices from the named dfb:: bindings generated for
// each kernel, so this header intentionally carries no numeric CB IDs.

#pragma once

constexpr uint32_t TILE_HW = 32;
constexpr uint32_t TILE_ELEMS = TILE_HW * TILE_HW;   // 1024
constexpr uint32_t TILE_SIZE_FP32 = TILE_ELEMS * 4;  // 4096 bytes
constexpr uint32_t TILE_SIZE_BF16 = TILE_ELEMS * 2;  // 2048 bytes
