// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common kernel functions used in data movement device kernels
// It's best to copy and paste the functions in rather than include the header as code size will likely explode
// Best to separate in to cpp/hpp at some point to avoid the code size explosion but need to figure out the linking
// issues

namespace tt::data_movement::common {

// this function is useful for converting bfloat16 values to float32
FORCE_INLINE float bfloat16_to_float32(uint16_t bfloat16_data) {
    uint32_t bits = static_cast<uint32_t>(bfloat16_data) << 16;

    // Extract the sign bit
    uint32_t sign = bits & 0x80000000;

    // Extract the exponent
    uint32_t exponent = bits & 0x7F800000;

    // Extract the mantissa
    uint32_t mantissa = bits & 0x007FFFFF;

    // Handle special cases
    if (exponent == 0 && mantissa == 0) {
        // Zero
        return sign ? -0.0f : 0.0f;
    } else if (exponent == 0x7F800000) {
        if (mantissa == 0) {
            // Infinity
            return sign ? -__builtin_huge_valf() : __builtin_huge_valf();
        } else {
            // NaN
            return __builtin_nanf("");
        }
    }

    // Assemble the float
    union {
        uint32_t u;
        float f;
    } ieee_float;

    ieee_float.u = sign | exponent | mantissa;
    return ieee_float.f;
}

FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, uint32_t val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

// Utility functions
template <uint32_t a, uint32_t b>
FORCE_INLINE constexpr uint32_t div_up() {
    static_assert(b > 0, "divisor must be greater than 0");
    return static_cast<uint32_t>((a + b - 1) / b);
}

template <uint32_t a, uint32_t b>
FORCE_INLINE constexpr uint32_t round_up() {
    return b * div_up<a, b>();
}
}  // namespace tt::data_movement::common
