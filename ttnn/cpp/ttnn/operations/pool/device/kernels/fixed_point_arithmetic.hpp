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

// Multiply two fixed-point numbers
ALWI constexpr int32_t fixed_mul(int32_t a, int32_t b) { return ((int64_t)a * b) >> FIXED_POINT_SHIFT; }

ALWI uint16_t float_to_bfloat16_non_constexpr(float val) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(&val);
    return uint16_t(*p >> 16);
}

// Convert fixed-point to bfloat16 for weight values
ALWI uint16_t fixed_to_bfloat16(int32_t fixed) {
    float fval = (float)fixed / FIXED_ONE;
    return float_to_bfloat16_non_constexpr(fval);
}

// Convert integer to fixed-point
ALWI constexpr int32_t int_to_fixed(int32_t i) { return i << FIXED_POINT_SHIFT; }

// Convert float to fixed-point
ALWI int32_t float_to_fixed(float f) { return static_cast<int32_t>(f * FIXED_ONE); }

// Extract integer part with rounding
ALWI constexpr int32_t fixed_to_int_round(int32_t fixed) { return (fixed + FIXED_HALF) >> FIXED_POINT_SHIFT; }

// Compute (a*b - c*d) >> shift + e
ALWI int32_t fixed_mul_sub_add(int32_t a, int32_t b, int32_t c, int32_t d, int32_t e) {
    int64_t prod1 = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    int64_t prod2 = static_cast<int64_t>(c) * static_cast<int64_t>(d);
    return static_cast<int32_t>(((prod1 - prod2) >> FIXED_POINT_SHIFT) + e);
}

// Compute (a*b + c*d) >> shift + e
ALWI int32_t fixed_mul_add_add(int32_t a, int32_t b, int32_t c, int32_t d, int32_t e) {
    int64_t prod1 = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    int64_t prod2 = static_cast<int64_t>(c) * static_cast<int64_t>(d);
    return static_cast<int32_t>(((prod1 + prod2) >> FIXED_POINT_SHIFT) + e);
}
