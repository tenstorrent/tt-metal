// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include <cstdint>
#include "sfpi.h"

using namespace sfpi;

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void _load_imm_(float imm) {
    vFloat const_v = s2vFloat16b(imm);
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = const_v;
        dst_reg++;
    }
}

inline void llk_math_eltwise_unary_load_imm(float val, uint dst_index, int vector_mode = (int)VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>([val]() { _load_imm_<APPROXIMATE>(val); }, dst_index, vector_mode);
}

inline void load_immediate_value(uint dst_index, float val) { MATH(llk_math_eltwise_unary_load_imm(val, dst_index)); }

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
