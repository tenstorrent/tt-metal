// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// batch_fft_common.h — Shared element/page geometry for device-side BATCH FFT.
// Metal 2.0 assigns DFB indices from the named dfb:: bindings generated for
// each kernel, so this header intentionally carries no numeric CB IDs.

#pragma once

#include <cstdint>

constexpr uint32_t TILE_HW = 32;
constexpr uint32_t TILE_ELEMS = TILE_HW * TILE_HW;   // 1024
constexpr uint32_t TILE_SIZE_FP32 = TILE_ELEMS * 4;  // 4096 bytes
constexpr uint32_t TILE_SIZE_BF16 = TILE_ELEMS * 2;  // 2048 bytes

// Converts an fp32 bit pattern to bf16 with round-to-nearest-even, matching the host-side
// bfloat16 constructor's default rounding (tt_metal/api/tt-metalium/bfloat16.hpp). Used by the
// FFT writer kernels' bf16 output path in place of plain truncation (>> 16), which biases every
// stored value toward zero (see issue #56532: 0.3-2.4% magnitude bias, 2-15x worse ulp error
// than a single RNE rounding at the end of the fp32 pipeline).
inline uint16_t fft_f32_bits_to_bf16_rne(uint32_t bits) {
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        // Inf/NaN: truncate but keep the quiet bit set for any NaN so it doesn't become Inf.
        return static_cast<uint16_t>((bits >> 16) | ((bits & 0x007fffffu) ? 0x0040u : 0u));
    }
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}
