// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Unsigned keys whose order is argmax's order (argmax_bfloat16_greater / argmax_float32_greater):
// key(a) > key(b) exactly when the comparator says a > b. So a scan keeps the first maximum with a
// plain integer compare.
//   - every NaN, of either sign and any payload, maps to one key above +inf, so the first NaN wins;
//   - +0.0 and -0.0 map to the same key;
//   - other values are ordered by sign and magnitude, negative magnitudes reversed.
// Kept free of device headers so the mapping can be checked on the host.

inline constexpr uint32_t ARGMAX_BF16_NAN_KEY = 0xFF81;  // one above key(+inf) = 0x8000 + 0x7F80
inline constexpr uint32_t ARGMAX_FP32_NAN_KEY = 0xFF800001u;

inline uint32_t argmax_bfloat16_key(uint32_t bits) {
    const uint32_t magnitude = bits & 0x7FFFu;
    if (magnitude > 0x7F80u) {
        return ARGMAX_BF16_NAN_KEY;
    }
    return (bits & 0x8000u) ? 0x8000u - magnitude : 0x8000u + magnitude;
}

inline uint32_t argmax_float32_key(uint32_t bits) {
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u) {
        return ARGMAX_FP32_NAN_KEY;
    }
    return (bits & 0x80000000u) ? 0x80000000u - magnitude : 0x80000000u + magnitude;
}
