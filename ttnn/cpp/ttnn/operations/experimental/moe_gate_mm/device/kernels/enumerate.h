// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "compute_kernel_api/common_globals.h"
enum MODE { ROWS, TILE };

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
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDR_MOD_0);
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 4, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDR_MOD_1);
}

inline void _enumerate_tile_(bool row_mode, float multiplier, float bias) {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    if (row_mode) {
        // load L10 to L4
        TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);
    } else {
        // bit shift L4 by 1 to divide by 2
        TTI_SFPSHFT(/* i12 */ 2, p_sfpu::LTILEID, p_sfpu::LREG4, 0);
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);
        // cast L4
    }

    TTI_SFPLOADI(p_sfpu::LREG6, InstrModLoadStore::FP16A, reinterpret_cast<uint16_t>(bias));
    TTI_SFPLOADI(p_sfpu::LREG7, InstrModLoadStore::FP16A, reinterpret_cast<uint16_t>(multiplier));

    float increment = row_mode ? 1 : 32;
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG4, 0);
    // L4 now has the enumerated value, we now only need to increment by the following
    TT_SFPLOADI(p_sfpu::LREG6, InstrModLoadStore::FP16A, reinterpret_cast<uint16_t>(increment));

    lltt::record<lltt::Exec>(0, 14);

    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG5, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG4, 0);
    TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG5, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG4, 0);
    TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG3, 0);

    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 2);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 16);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_1, 18);

    TTI_SFPTRANSP(0, 0, 0, 0);

    for (uint32_t i = 0; i < 7; i++) {
        lltt::replay(0, 14);
    }
}
inline void _llk_math_enumerate_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(
        ckernel::sfpu::_enumerate_tile_configure_addrmod_);
}

inline void _llk_math_enumerate_tile_(uint32_t dst_index, bool row_mode, float multiplier, float bias) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_enumerate_tile_, dst_index, VectorMode::RC_custom, row_mode, multiplier, bias);
}
#endif

inline void enumerate_tile_init() { MATH((_llk_math_enumerate_tile_init_())); }

ALWI void enumerate_tile(uint32_t dst_index, bool row_mode, float multiplier = 1, float bias = 0) {
    MATH((_llk_math_enumerate_tile_(dst_index, row_mode, multiplier, bias)));
}
}

}  // namespace ckernel
