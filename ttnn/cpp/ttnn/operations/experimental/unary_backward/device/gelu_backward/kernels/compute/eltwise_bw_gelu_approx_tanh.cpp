// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_shift.h"
#include "compute_kernel_api.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

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
static void load_immediate_value(uint dst_index) {
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

static void copy_value(uint dst_index0, uint32_t dst_index1) {
    MATH(llk_math_eltwise_binary_sfpu_copy_value(dst_index0, dst_index1));
}

namespace NAMESPACE {
using namespace sfpi;
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // grad
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // x
    constexpr auto cb_out0 = tt::CBIndex::c_2;  // out

    constexpr uint32_t bits_1p0 = 0x3f800000;             // 1.0f
    constexpr uint32_t bits_0p5 = 0x3F000000;             // 0.5f
    constexpr uint32_t bits_sqrt_2_over_pi = 0x3f4c422a;  // sqrt(2/pi)
    constexpr uint32_t bits_0p044715 = 0x3d372713;        // 0.044715
    constexpr uint32_t bits_0p134145 = 0x3e095d4f;        // 0.134145

    unary_op_init_common(cb_in0, cb_out0);
    add_binary_tile_init();
    mul_binary_tile_init();
    square_tile_init();
    tanh_tile_init();
    sub_binary_tile_init();

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();
        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in0, i, 0);
            copy_tile(cb_in1, i, 1);

            // tile[2] = x
            copy_value(2, 1);

            // tile[1] = x^3
            square_tile(1);
            mul_binary_tile(1, 2);

            // tile[1] = 0.044715 * x^3
            load_immediate_value<bits_0p044715>(3);
            mul_binary_tile(1, 3);

            // tile[1] = x + 0.044715 * x^3
            add_binary_tile(1, 2);

            // tile[1] = sqrt(2/π) * (x + 0.044715 * x^3)
            load_immediate_value<bits_sqrt_2_over_pi>(3);
            mul_binary_tile(1, 3);

            // tile[1] = tanh(sqrt(2/π) * (x + 0.044715 * x^3))
            tanh_tile(1);
            copy_value(7, 1);  // save tanh to tile[7]

            // CDF term: tile[1] = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            load_immediate_value<bits_1p0>(3);
            add_binary_tile(1, 3);
            load_immediate_value<bits_0p5>(3);
            mul_binary_tile(1, 3);

            // tile[7] = 1 - tanh^2
            square_tile(7);
            load_immediate_value<bits_1p0>(6);
            sub_binary_tile(6, 7);
            copy_value(7, 6);

            // tile[5] = (1 + 0.134145 * x**2)
            load_immediate_value<bits_0p134145>(6);
            copy_value(5, 2);
            square_tile(5);         // x^2
            mul_binary_tile(5, 6);  // 0.134145 * x**2
            load_immediate_value<bits_1p0>(6);
            add_binary_tile(5, 6);  // 1 + 0.134145 * x**2

            // PDF term: tile[5] = 0.5 * sqrt(2/π) * (1 + 0.134145 * x^2) * (1 - tanh^2)
            mul_binary_tile(5, 7);
            load_immediate_value<bits_sqrt_2_over_pi>(6);
            mul_binary_tile(5, 6);
            load_immediate_value<bits_0p5>(6);
            mul_binary_tile(5, 6);

            // tile[5] = x * pdf tern
            copy_value(6, 2);
            mul_binary_tile(5, 6);

            // result: tile[1] = grad * (cdf_term + x * pdf_term)
            add_binary_tile(1, 5);  // cdf_term + x * pdf_term
            // tile[0] = grad * (cdf_term + x * pdf_term)
            mul_binary_tile(0, 1);

            pack_tile(0, cb_out0);
        }
        tile_regs_commit();
        tile_regs_release();

        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
}  // namespace NAMESPACE
