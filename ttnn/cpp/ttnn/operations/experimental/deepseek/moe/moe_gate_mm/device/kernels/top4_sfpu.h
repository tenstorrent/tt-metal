// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

/**
 * @brief Selects the top-4 values from 8 inputs and produces a bitmask of their indices.
 *
 * Loads 8 FP16B values (from the lower face of the DST tile at `dst_index`) into
 * LREG0-7, transposes so each LREG holds one core's data across 32 lanes, then
 * fuses a 3-bit core index (0-7) into the lower 16 bits of each register.
 * Uses a **bubble sort** network (4 passes of 7, 6, 5, 4 adjacent SFPSWAPs) to
 * move the 4 largest values into LREG0-3. The core indices of the top-4 winners
 * are extracted and OR'd into a single 8-bit bitmask per lane via shift-and-OR.
 *
 * Input:  8 FP16B values per lane, read from the lower face (rows 0-7) of the DST
 *         tile at offset 32 (face 1).
 * Output: One uint16 bitmask per lane (bits 0-7, one per core), written to row 0
 *         of face 0 of the same DST tile (LO16 mode).
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

inline void _top4_configure_addrmod_() {
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
        .dest = {.incr = 6, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_2);

    addr_mod_t{
        .dest = {.incr = -6, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_3);
}

inline void _calculate_top4_() {
    // Reset Dst RWC to 0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    //-------------------------------------------------------------------------
    // Step 1: Load BF16 values to HI 16 bits from offsets 0, 2, 8, 10 -> 4-3
    //-------------------------------------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 32);  // offset 0
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 32);  // offset 2
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 32);  // offset 8
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 32);  // offset 10

    //-------------------------------------------------------------------------
    // Step 2: Load BF16 values to HI 16 bits from offsets 4, 6, 12, 14 -> LREG4-7
    //-------------------------------------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 32);  // offset 4
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 32);  // offset 6
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 32);  // offset 12
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_3, 32);  // offset 14

    //-------------------------------------------------------------------------
    // Step 3: Transpose
    // After transpose: each LREG has 32 lanes with data from 32 different tokens
    // Now all lanes in LREG0 are from core 0, LREG1 from core 1, etc.
    //-------------------------------------------------------------------------
    TTI_SFPTRANSP(0, 0, 0, 0);

    //-------------------------------------------------------------------------
    // Step 4: Fuse core indices (0-7) into LO 16 bits of each LREG
    //-------------------------------------------------------------------------
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0);  // core index 0
    TTI_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 1);  // core index 1
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 2);  // core index 2
    TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 3);  // core index 3
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, 4);  // core index 4
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, 5);  // core index 5
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, 6);  // core index 6
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, 7);  // core index 7

    //-------------------------------------------------------------------------
    // Step 5: Bubble sort to extract top 4 values into LREG0-3
    // Using ALL_ROWS_MAX SWAP to keep max values in first register
    // Bubble from right to left to move maximum values towards LREG0
    // After sorting: LREG0 = max, LREG1 = 2nd, LREG2 = 3rd, LREG3 = 4th
    //-------------------------------------------------------------------------

    // Pass 1: Bubble maximum value to LREG0 (7 comparisons, right to left)
    lltt::record<lltt::Exec>(0, 7);
    TTI_SFPSWAP(0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

    // Pass 2: Bubble 2nd maximum to LREG1 (6 comparisons, right to left)
    lltt::replay(0, 6);

    // Pass 3: Bubble 3rd maximum to LREG2 (5 comparisons, right to left)
    lltt::replay(0, 5);

    // Pass 4: Bubble 4th maximum to LREG3 (4 comparisons, right to left)
    lltt::replay(0, 4);

    // Result: LREG0-3 contain the top 4 values (with their indices in LO 16 bits)
    //-------------------------------------------------------------------------
    // Step 8: Generate a bitmask vector with the top 4 indices
    //-------------------------------------------------------------------------
    // Initialize accumulator LREG4 to 0
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_USHORT, 0);

    for (int i = 0; i < 4; i++) {
        // Load mask 0xFF to extract LO 8 bits (only the indices)
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0xFF);

        // Extract LO 16 bits (the indices) from LREG[i]
        TTI_SFPAND(0, p_sfpu::LREG0 + i, p_sfpu::LREG7, 0);
        TTI_SFPNOP;

        // Load value 1 into LREG5 (for bitmask generation)
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 1);
        TTI_SFPNOP;

        // Shift left by the value in LREG[i]: LREG6 = (1 << LREG[i])
        // This creates a bitmask where bit LREG[i] is set
        // SFPSHFT2 mode 5 = SFPSHFT2_MOD1_SHFT_LREG (variable shift by LREG)
        TTI_SFPSHFT2(p_sfpu::LREG5, p_sfpu::LREG7, p_sfpu::LREG6, SFPSHFT2_MOD1_SHFT_LREG);
        TTI_SFPNOP;

        // OR into accumulator: LREG4 |= LREG6
        TTI_SFPOR(0, p_sfpu::LREG6, p_sfpu::LREG4, 0);
    }

    //-------------------------------------------------------------------------
    // Step 9: Write back the 4 indices as raw uint16 values
    // Using LO16 mode to write the lower 16 bits (the indices)
    //-------------------------------------------------------------------------
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, 0);  // Write 1st index
}

}  // namespace sfpu

inline void _llk_math_top4_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(ckernel::sfpu::_top4_configure_addrmod_);
}

inline void _llk_math_top4_tile_(uint32_t dst_index) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_calculate_top4_, dst_index, VectorMode::RC_custom);
}

#endif

/**
 * @brief Initializes the top-4 selection SFPU operation.
 */
inline void top4_tile_init() { MATH((_llk_math_top4_tile_init_())); }

ALWI void top4_tile(uint32_t dst_index) { MATH((_llk_math_top4_tile_(dst_index))); }

}  // namespace ckernel
