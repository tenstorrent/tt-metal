// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

namespace ckernel {

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "lltt.h"
#include "sfpi.h"

namespace sfpu {

inline void _enumerate_tile_configure_addrmod_() {
    // TODO: No idea why we need this offset only when programming, but it works.
    constexpr uint32_t ADDRMOD_OFFSET = 4;

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_0);
}

inline void _enumerate_tile_(bool row_mode, float multiplier, float bias) {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    if (row_mode) {
        // load L10 to L4
        TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);
    } else {
        TTI_SFPMOV(0, p_sfpu::LTILEID, p_sfpu::LREG4, 0);
        // bit shift L4 by 1 to divide by 2
        //  TTI_SFPSHFT(/* i12 */ (-1) & 0xfff, 0, p_sfpu::LREG4, 1);
        // cast L4
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, /*MOD=IMM1*/ 1);
    }

    union {
        float f;
        uint32_t u;
    } bias_bits = {.f = bias};
    TT_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, bias_bits.u >> 16);

    union {
        float f;
        uint32_t u;
    } multiplier_bits = {.f = multiplier};
    TT_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, multiplier_bits.u >> 16);

    // Work on multiplier and bias
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG4, 0);

    float row_increment = (row_mode ? 1 : 64) * multiplier;
    union {
        float f;
        uint32_t u;
    } row_increment_bits = {.f = row_increment};

    // L4 now has the enumerated value, we now only need to row_increment by the following
    TT_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, row_increment_bits.u >> 16);

    float face_increment = (row_mode ? 1 : 512) * multiplier;
    union {
        float f;
        uint32_t u;
    } face_increment_bits = {.f = face_increment};
    TT_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, face_increment_bits.u >> 16);

    // Let us make L7 have the values for the odd columns
    auto mul_reg = (multiplier < 0.0f) ? p_sfpu::LCONST_neg1 : p_sfpu::LCONST_1;
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_1, mul_reg, p_sfpu::LREG7, 0);

    lltt::record<lltt::NoExec>(0, 12);

    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    TTI_NOP;
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    TTI_NOP;
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    TTI_NOP;
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    TTI_NOP;
    // TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG4, 0);
    // TTI_SFPMOV(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);
    // TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::LREG3, 0);
    // TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG7, 0);

    for (uint32_t half = 0; half < 2; half++) {
        for (uint32_t row = 0; row < 4; row++) {
            lltt::replay(0, 12);
            TTI_SFPTRANSP(0, 0, 0, 0);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, half * 32 + row * 4 + 0);
            TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, half * 32 + row * 4 + 2);
            TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, half * 32 + row * 4 + 16);
            TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, half * 32 + row * 4 + 18);
            TTI_SFPTRANSP(0, 0, 0, 0);
        }
    }
}
}  // namespace sfpu

inline void _llk_math_enumerate_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(
        ckernel::sfpu::_enumerate_tile_configure_addrmod_);
}

inline void _llk_math_enumerate_tile_(uint32_t dst_index, bool row_mode, float multiplier, float bias) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_enumerate_tile_, dst_index, VectorMode::RC_custom, row_mode, multiplier, bias);
}

#endif

/**
 * @brief Initializes the enumerated tile calculation.
 */
inline void enumerate_tile_init() { MATH((_llk_math_enumerate_tile_init_())); }

/**
 * @brief Fills the destination tile with the enumerated values.
 * @param tile_index The tile index
 * @param dst_index The destination tile index
 */
ALWI void enumerate_tile(uint32_t dst_index, bool row_mode, float multiplier, float bias) {
    MATH((_llk_math_enumerate_tile_(dst_index, row_mode, multiplier, bias)));
}

}  // namespace ckernel
