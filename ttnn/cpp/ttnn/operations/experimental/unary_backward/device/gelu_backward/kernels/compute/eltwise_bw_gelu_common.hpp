// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include <cstdint>
#include "sfpi.h"

using namespace sfpi;

// TODO: This conversation is obsolete. Remove it.
template <uint32_t Bits>
constexpr float asFloat() {
    union {
        uint32_t i;
        float f;
    } u{Bits};
    return u.f;
}

template <bool APPROXIMATE, uint32_t IMM_BITS, int ITERATIONS = 8>
inline void _load_imm_() {
    float imm = asFloat<IMM_BITS>();
    vFloat const_v = s2vFloat16a(imm);
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = const_v;
        dst_reg++;
    }
}

template <uint32_t IMM_BITS>
inline void llk_math_eltwise_unary_load_imm(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(_load_imm_<APPROXIMATE, IMM_BITS>, dst_index, vector_mode);
}

template <uint32_t IMM_BITS>
inline void load_immediate_value(uint dst_index) {
    MATH(llk_math_eltwise_unary_load_imm<IMM_BITS>(dst_index));
}

template <bool APPROXIMATION_MODE, bool small_to_big_index, int ITERATIONS = 8>
inline void _copy_value_(const uint dst_offset) {
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 32;
        if constexpr (small_to_big_index) {
            dst_reg[dst_offset * dst_tile_size] = dst_reg[0];
        } else {
            dst_reg[0] = dst_reg[dst_offset * dst_tile_size];
        }
        dst_reg++;
    }
}

inline void llk_math_eltwise_binary_sfpu_copy_value(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    if (dst_index0 > dst_index1) {
        llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
            _copy_value_<APPROXIMATE, 1>, dst_index0, dst_index1, vector_mode);
    } else {
        llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
            _copy_value_<APPROXIMATE, 0>, dst_index0, dst_index1, vector_mode);
    }
}

inline void copy_value(uint dst_index0, uint32_t dst_index1) {
    MATH(llk_math_eltwise_binary_sfpu_copy_value(dst_index0, dst_index1));
}
