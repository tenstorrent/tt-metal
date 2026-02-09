// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ALWI inline __attribute__((always_inline))

namespace fixed_point_arithmetic {

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
//
// Converts a Q16.16 fixed-point value to bfloat16 format by:
// 1. Finding the most significant bit position to determine the magnitude
// 2. Computing the IEEE 754 exponent (biased by 127, adjusted for Q16.16 format)
// 3. Extracting/shifting the mantissa to fit the 23-bit float mantissa
// 4. Truncating to bfloat16 by taking the upper 16 bits (sign + 8-bit exp + 7-bit mantissa)
//
// Special cases:
// - Zero maps to 0x0000
// - FIXED_ONE (1.0) maps to 0x3F80 (bfloat16 representation of 1.0)
// - Underflow (exponent <= 0) returns 0x0000
// - Overflow (exponent >= 255) returns 0x7F80 (infinity)
ALWI constexpr uint16_t fixed_to_bf16(int32_t fixed_val) {
    if (fixed_val == 0) {
        return 0x0000;
    }
    if (fixed_val == FIXED_ONE) {
        return 0x3F80;
    }

    uint32_t fixed_bits = static_cast<uint32_t>(fixed_val);

    int most_significant_bit_pos = 0;
    for (int bit_index = 31; bit_index >= 0; --bit_index) {
        if (fixed_bits & (1u << bit_index)) {
            most_significant_bit_pos = bit_index;
            break;
        }
    }

    int ieee754_exponent = 127 + (most_significant_bit_pos - FIXED_POINT_SHIFT);
    if (ieee754_exponent <= 0) {
        return 0x0000;
    }
    if (ieee754_exponent >= 255) {
        return 0x7F80;
    }

    int mantissa_shift_amount = most_significant_bit_pos - 23;
    uint32_t float_mantissa = 0;
    if (mantissa_shift_amount > 0) {
        float_mantissa = (fixed_bits >> mantissa_shift_amount) & 0x7FFFFF;
    } else if (mantissa_shift_amount < 0) {
        float_mantissa = (fixed_bits << (-mantissa_shift_amount)) & 0x7FFFFF;
    } else {
        float_mantissa = fixed_bits & 0x7FFFFF;
    }

    uint32_t float32_bits = (static_cast<uint32_t>(ieee754_exponent) << 23) | float_mantissa;

    return static_cast<uint16_t>(float32_bits >> 16);
}

}  // namespace fixed_point_arithmetic
