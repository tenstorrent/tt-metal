// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#define ALWI inline __attribute__((always_inline))

constexpr int32_t Q16_ONE = 1 << 16;
constexpr int32_t Q16_HALF = 1 << 15;

using fixed_point_t = int32_t;

ALWI fixed_point_t int_to_q16(int32_t i) { return i << 16; }

ALWI fixed_point_t float_to_q16(float f) { return static_cast<fixed_point_t>(f * 65536.0f); }

ALWI int32_t q16_to_int(fixed_point_t q) { return q >> 16; }

ALWI int32_t q16_to_int_round(fixed_point_t q) { return (q + Q16_HALF) >> 16; }

ALWI fixed_point_t q16_mul(fixed_point_t a, fixed_point_t b) {
    int64_t result = (static_cast<int64_t>(a) * static_cast<int64_t>(b)) >> 16;
    return static_cast<fixed_point_t>(result);
}

ALWI fixed_point_t q16_add(fixed_point_t a, fixed_point_t b) { return a + b; }

ALWI fixed_point_t q16_sub(fixed_point_t a, fixed_point_t b) { return a - b; }

ALWI bool is_coordinate_valid(int32_t coord, uint32_t max_val) {
    return coord >= 0 && coord < static_cast<int32_t>(max_val);
}

ALWI fixed_point_t
q16_mul_sub_add(fixed_point_t a, fixed_point_t b, fixed_point_t c, fixed_point_t d, fixed_point_t e) {
    int64_t prod1 = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    int64_t prod2 = static_cast<int64_t>(c) * static_cast<int64_t>(d);
    int64_t result = ((prod1 - prod2) >> 16) + e;
    return static_cast<fixed_point_t>(result);
}

ALWI fixed_point_t
q16_mul_add_add(fixed_point_t a, fixed_point_t b, fixed_point_t c, fixed_point_t d, fixed_point_t e) {
    int64_t prod1 = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    int64_t prod2 = static_cast<int64_t>(c) * static_cast<int64_t>(d);
    int64_t result = ((prod1 + prod2) >> 16) + e;
    return static_cast<fixed_point_t>(result);
}

#undef ALWI
