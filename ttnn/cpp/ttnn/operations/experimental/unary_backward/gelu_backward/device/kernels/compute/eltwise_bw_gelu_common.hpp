// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include <cstdint>
#include "sfpi.h"

using namespace sfpi;

namespace {
template <bool APPROXIMATE, int ITERATIONS = 8>
void _load_imm_(float imm) {
    vFloat const_v = s2vFloat16b(imm);
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = const_v;
        dst_reg++;
    }
}

void llk_math_eltwise_unary_load_imm(float val, uint dst_index, int vector_mode = (int)VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>([val]() { _load_imm_<APPROXIMATE>(val); }, dst_index, vector_mode);
}

void _llk_math_load_imm_sfpu_init_() {
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, bool small_to_big_index, int ITERATIONS = 8>
void _copy_value_(const uint dst_offset) {
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

void llk_math_eltwise_binary_sfpu_copy_value(uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    if (dst_index0 > dst_index1) {
        llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
            _copy_value_<APPROXIMATE, 1>, dst_index0, dst_index1, vector_mode);
    } else {
        llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
            _copy_value_<APPROXIMATE, 0>, dst_index0, dst_index1, vector_mode);
    }
}

}  // namespace

inline void load_immediate_value_init() { MATH(_llk_math_load_imm_sfpu_init_()); }

// Load immediate value (val) to the destination register dst_index
inline void load_immediate_value(uint dst_index, float val) { MATH(llk_math_eltwise_unary_load_imm(val, dst_index)); }

// Copy value from dst_index1 destination register to dst_index0
inline void copy_value(uint dst_index0, uint32_t dst_index1) {
    MATH(llk_math_eltwise_binary_sfpu_copy_value(dst_index0, dst_index1));
}
