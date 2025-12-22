// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ALWI inline __attribute__((always_inline))

// Fixed-point math constants and helpers for Q16.16 format
constexpr int32_t FIXED_POINT_SHIFT = 16;
constexpr int32_t FIXED_ONE = 1 << FIXED_POINT_SHIFT;         // 1.0 in Q16.16
constexpr int32_t FIXED_HALF = 1 << (FIXED_POINT_SHIFT - 1);  // 0.5 in Q16.16

// Extract integer part from fixed-point
ALWI constexpr int32_t fixed_to_int(int32_t fixed) { return fixed >> FIXED_POINT_SHIFT; }

// Extract fractional part from fixed-point (0 to FIXED_ONE)
ALWI constexpr int32_t fixed_frac(int32_t fixed) { return fixed & ((1 << FIXED_POINT_SHIFT) - 1); }

// Multiply two fixed-point numbers (works in both constexpr and runtime contexts)
ALWI constexpr int32_t fixed_mul(int32_t a, int32_t b) {
    return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> FIXED_POINT_SHIFT);
}

// Convert fixed-point to bfloat16 (works in both constexpr and runtime contexts)
ALWI constexpr uint16_t fixed_to_bf16(int32_t fixed_val) {
    if (fixed_val == 0) {
        return 0x0000;
    }
    if (fixed_val == FIXED_ONE) {
        return 0x3F80;
    }

    uint32_t val = static_cast<uint32_t>(fixed_val);

    int msb_pos = 0;
    for (int i = 31; i >= 0; --i) {
        if (val & (1u << i)) {
            msb_pos = i;
            break;
        }
    }

    int exponent = 127 + (msb_pos - FIXED_POINT_SHIFT);
    if (exponent <= 0) {
        return 0x0000;
    }
    if (exponent >= 255) {
        return 0x7F80;
    }

    int mantissa_shift = msb_pos - 23;
    uint32_t mantissa = 0;
    if (mantissa_shift > 0) {
        mantissa = (val >> mantissa_shift) & 0x7FFFFF;
    } else if (mantissa_shift < 0) {
        mantissa = (val << (-mantissa_shift)) & 0x7FFFFF;
    } else {
        mantissa = val & 0x7FFFFF;
    }

    uint32_t float_bits = (static_cast<uint32_t>(exponent) << 23) | mantissa;

    return static_cast<uint16_t>(float_bits >> 16);
}
