// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

/**
 * @brief Computes the sum of the two largest values in a tile (per lane).
 *
 * Reads all 32 FP16B values (8 groups of 4 rows) from the destination tile at
 * `dst_index` and finds the top-2 maximum values using a tournament-style
 * compare-and-swap network. Each group of 4 values is reduced to a local top-2
 * via 5 SFPSWAP operations, then merged with the running top-2 via 3 SFPSWAPs.
 * After processing all 8 groups, the two winners are summed (LREG0 += LREG1).
 *
 * Input:  32 FP16B values per lane, read in-place from the DST tile (8 groups x 4 rows,
 *         upper and lower faces).
 * Output: A single FP16B sum-of-top-2 value per lane, written to the first 4 rows
 *         (rows 0, 2, 4, 6) of face 0 of the same DST tile. All 4 rows contain the
 *         same replicated result.
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

inline void _top2_configure_addrmod_() {
    // SFPU configuration shifts to use ADDRMOD4-7.
    constexpr uint32_t ADDRMOD_OFFSET = 4;

    addr_mod_t{
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_0);

    addr_mod_t{
        .dest = {.incr = 2, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_1);

    addr_mod_t{
        .dest = {.incr = 14, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_2);

    addr_mod_t{
        .dest = {.incr = -14, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_3);
}

inline void _top2_calculate_top2_() {
    constexpr uint16_t NEG_INF_BF16 = 0xFF80;

    // Reset Dst RWC to 0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Initialize LREG0/1 with -infinity
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, NEG_INF_BF16);

    //-------------------------------------------------------------------------
    // Group 0
    lltt::record<lltt::Exec>(0, 14);

    TTI_SFPTRANSP(0, 0, 0, 0);
    // Load 4 tile rows into LREG4-7
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);
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
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);

    //-------------------------------------------------------------------------
    lltt::replay(0, 14);  // Group 1
    lltt::replay(0, 14);  // Group 2
    lltt::replay(0, 14);  // Group 3
    //-------------------------------------------------------------------------
    // We are now reading in bottom two faces
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);

    lltt::record<lltt::Exec>(14, 6);

    // Group 4
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
 * @brief Initializes the sum-of-top-2 SFPU operation.
 */
inline void sum_top2_tile_init() { MATH((_llk_math_sum_top2_tile_init_())); }

ALWI void sum_top2_tile(uint32_t dst_index) { MATH((_llk_math_sum_top2_tile_(dst_index))); }

}  // namespace ckernel
