// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#define ALWI inline __attribute__((always_inline))

// Q16.16 Fixed-Point Arithmetic Helper Functions
//
// Q16.16 format uses 32-bit signed integers where:
// - Upper 16 bits: integer part (range: -32768 to 32767)
// - Lower 16 bits: fractional part (precision: 1/65536 ≈ 0.0000152588)
//
// This provides sufficient precision for coordinate transformations and
// bilinear interpolation weights in image rotation operations.

// ============================================================================
// Constants
// ============================================================================

constexpr int32_t Q16_ONE = 1 << 16;  // 1.0 in Q16.16 format (65536)

// ============================================================================
// Basic Conversion Functions
// ============================================================================

// Convert float to Q16.16 fixed-point
ALWI int32_t float_to_q16(float f) { return static_cast<int32_t>(f * 65536.0f); }

// Convert Q16.16 to float (for debugging/verification)
ALWI float q16_to_float(int32_t q) { return static_cast<float>(q) / 65536.0f; }

// Convert integer to Q16.16
ALWI int32_t int_to_q16(int32_t i) { return i << 16; }

// ============================================================================
// Arithmetic Operations
// ============================================================================

// Multiply two Q16.16 numbers
// Uses 64-bit intermediate to prevent overflow
ALWI int32_t q16_mul(int32_t a, int32_t b) {
    int64_t result = (static_cast<int64_t>(a) * static_cast<int64_t>(b)) >> 16;
    return static_cast<int32_t>(result);
}

// Add two Q16.16 numbers (no special handling needed)
ALWI int32_t q16_add(int32_t a, int32_t b) { return a + b; }

// Subtract two Q16.16 numbers (no special handling needed)
ALWI int32_t q16_sub(int32_t a, int32_t b) { return a - b; }

// ============================================================================
// Format Conversions
// ============================================================================

// Convert Q16.16 to plain int32 (floor operation, extracts integer part)
// Returns: Plain int32 value (NOT in Q16.16 format)
ALWI int32_t q16_to_int(int32_t q) {
    return q >> 16;  // Extract upper 16 bits
}

// ============================================================================
// Q16.16 Operations (input and output stay in Q16.16 format)
// ============================================================================

// Extract fractional part from Q16.16 (returns value in [0, 1) range, still Q16.16 format)
// Returns: Q16.16 value with only fractional bits (integer part cleared)
ALWI int32_t q16_frac(int32_t q) {
    return q & 0xFFFF;  // Keep only lower 16 bits (fractional part)
}

// Compute (1.0 - x) for Q16.16 number x in [0, 1] range
// Returns: Q16.16 value
ALWI int32_t q16_one_minus(int32_t q_frac) { return Q16_ONE - q_frac; }

#undef ALWI
