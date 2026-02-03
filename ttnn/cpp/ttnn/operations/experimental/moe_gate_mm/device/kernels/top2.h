// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "compute_kernel_api/common_globals.h"

namespace ckernel {

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "lltt.h"
#include "sfpi.h"

namespace sfpu {

inline void _top2_configure_addrmod_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_1);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 14},
    }
        .set(ADDR_MOD_2);

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = -14},
    }
        .set(ADDR_MOD_3);
}

inline void _top2_calculate_top2_() {
    constexpr uint32_t NEG_INF_FP32 = 0xFF800000;

    // Reset Dst RWC to 0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Initialize LREG0/1 with -infinity
    TT_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, NEG_INF_FP32 >> 16);
    TT_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, NEG_INF_FP32 & 0xFFFF);

    TT_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, NEG_INF_FP32 >> 16);
    TT_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, NEG_INF_FP32 & 0xFFFF);

    //-------------------------------------------------------------------------
    // Group 0
    lltt::record<lltt::Exec>(0, 14);

    TTI_SFPTRANSP(0, 0, 0, 0);
    // Load 4 tile rows into LREG4-7
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 16);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 18);
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Tournament on LREG4-7 to find top-2 of this batch
    // After: LREG4 = max, LREG5 = 2nd max
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);

    // Merge batch top-2 (LREG4/5) with running top-2 (LREG0/1)
    // After: LREG0 = max, LREG1 = 2nd max
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);

    //-------------------------------------------------------------------------
    lltt::replay(0, 14);  // Group 1
    lltt::replay(0, 14);  // Group 2
    lltt::replay(0, 14);  // Group 3
    //-------------------------------------------------------------------------
    // Group 4
    lltt::record<lltt::Exec>(14, 6);

    TTI_SFPTRANSP(0, 0, 0, 0);
    // Load 4 tile rows into LREG4-7
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 32);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 32);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 32);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 32);
    TTI_SFPTRANSP(0, 0, 0, 0);

    lltt::replay(6, 8);

    // Group 5
    lltt::replay(14, 6);
    lltt::replay(6, 8);

    // Group 6
    lltt::replay(14, 6);
    lltt::replay(6, 8);

    // Group 7
    lltt::replay(14, 6);
    lltt::replay(6, 8);

    // Store Results
    // Sum of top-2: LREG0 = LREG0 + LREG1
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPNOP;

    // Prepare LREG0-3 for transpose-back (all with sum values)
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 2);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 4);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, 6);
}

}  // namespace sfpu

inline void _llk_math_sum_top2_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(ckernel::sfpu::_top2_configure_addrmod_);
}

inline void _llk_math_sum_top2_tile_(uint32_t dst_index) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_top2_calculate_top2_, dst_index, VectorMode::RC_custom);
}

#endif

/**
 * @brief Initializes the sum of the top-2 of the input tile.
 * @return None. Modifies the first two rows of the first face of the tile with the result.
 */
inline void sum_top2_tile_init() { MATH((_llk_math_sum_top2_tile_init_())); }

/**
 * @brief Calculates the sum of the top-2 of the input tile at dst index 0.
 * @return None. Modifies the first two rows of the first face of the tile with the result.
 */
ALWI void sum_top2_tile(uint32_t dst_index) { MATH((_llk_math_sum_top2_tile_(dst_index))); }

}  // namespace ckernel
