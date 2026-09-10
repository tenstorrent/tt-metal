// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// fft_inner_host.hpp — Tile constants and arithmetic helpers shared by the
//                      FFT ProgramSpec factories.
//
// Provides:
//   * kTileHW / kTileElems / kTileSizeFp32 — Tensix tile geometry
//   * log2u() / bit_rev() — tiny arithmetic helpers used by twiddle math
//
// Device storage is represented by ttnn::Tensor and bound through named
// TensorParameters. This header deliberately contains no legacy MeshBuffer
// allocation or raw-address helpers.

#pragma once

#include <cstdint>

namespace fft_example {

constexpr uint32_t kTileHW = 32;
constexpr uint32_t kTileElems = kTileHW * kTileHW;              // 1024
constexpr uint32_t kTileSizeFp32 = kTileElems * sizeof(float);  // 4096 bytes

inline uint32_t log2u(uint32_t x) {
    uint32_t r = 0;
    while ((1u << r) < x) {
        ++r;
    }
    return r;
}

inline uint32_t bit_rev(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

}  // namespace fft_example
