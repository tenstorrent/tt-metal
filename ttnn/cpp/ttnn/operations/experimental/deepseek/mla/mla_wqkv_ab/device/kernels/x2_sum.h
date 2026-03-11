// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

/**
 * @brief Computes sum(x^2) over 8 groups of 4 rows for each tile.
 *
 * For each tile:
 *  - Initialize accumulator LREG4 to zero
 *  - Load each group into LREG0-3
 *  - Square each value (x * x)
 *  - Accumulate into LREG4
 * The final accumulator is written to the destination tile.
 */

namespace ckernel {

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "lltt.h"
#include "sfpi.h"

namespace sfpu {

inline void _x2_sum_configure_addrmod_() {
    // SFPU configuration shifts to use ADDRMOD4-7.
    constexpr uint32_t ADDRMOD_OFFSET = 4;

    addr_mod_t{
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_0);

    addr_mod_t{
        .dest = {.incr = 4, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_1);

    addr_mod_t{
        .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_2);

    addr_mod_t{
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_3);
}

inline void _x2_sum_calculate_tile_(uint32_t num_tiles) {
    // Reset Dst RWC to 0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Running accumulator for sum(x^2)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);

    //-------------------------------------------------------------------------
    // Face 0, Face 1: groups 0..3
    //-------------------------------------------------------------------------
    lltt::record<lltt::Exec>(0, 18);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 2);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 16);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_1, 18);

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);

    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    lltt::replay(0, 18);  // Group 1
    lltt::replay(0, 18);  // Group 2
    lltt::replay(0, 18);  // Group 3

    //-------------------------------------------------------------------------
    // Face 2, Face 3: groups 4..7
    //-------------------------------------------------------------------------
    lltt::record<lltt::Exec>(18, 2);
    // Move dst by 16, use two pseudo NOPs
    TTI_SFPLOAD(p_sfpu::LCONST_1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);
    TTI_SFPLOAD(p_sfpu::LCONST_1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);

    lltt::replay(0, 18);  // Group 4
    lltt::replay(0, 18);  // Group 5
    lltt::replay(0, 18);  // Group 6
    lltt::replay(0, 18);  // Group 7

    lltt::replay(18, 2);

    // Repeat this for all tiles
    for (uint32_t tile_index = 1; tile_index < num_tiles; tile_index++) {
        lltt::replay(0, 18);  // Group 0
        lltt::replay(0, 18);  // Group 1
        lltt::replay(0, 18);  // Group 2
        lltt::replay(0, 18);  // Group 3

        lltt::replay(18, 2);

        lltt::replay(0, 18);  // Group 4
        lltt::replay(0, 18);  // Group 5
        lltt::replay(0, 18);  // Group 6
        lltt::replay(0, 18);  // Group 7

        lltt::replay(18, 2);
    }

    // Write output to destination tile
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, 64 * 7 + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, 64 * 7 + 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, 64 * 7 + 4);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 64 * 7 + 6);
}

}  // namespace sfpu

inline void _llk_math_x2_sum_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(ckernel::sfpu::_x2_sum_configure_addrmod_);
}

inline void _llk_math_x2_sum_single_tile_(uint32_t dst_index, uint32_t num_tiles) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_x2_sum_calculate_tile_, dst_index, VectorMode::RC_custom, num_tiles);
}

#endif

/**
 * @brief Initializes the X^2 sum SFPU operation.
 */
inline void x2_sum_init() { MATH((_llk_math_x2_sum_tile_init_())); }

/**
 * @brief Runs X^2 reduction over `num_tiles`; final result is on last tile.
 */
ALWI void x2_sum_tile(uint32_t dst_index, uint32_t num_tiles) {
    MATH((_llk_math_x2_sum_single_tile_(dst_index, num_tiles)));
}

}  // namespace ckernel
