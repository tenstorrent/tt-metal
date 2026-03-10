// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

/**
 * @brief Applies Rotary Position Embedding (RoPE) to Q/K/V tiles.
 *
 * Placeholder SFPU for RoPE application. The _calculate_rope_ body is hollow;
 * addrmod config and structure are retained for future implementation.
 *
 * Input:  Q/K/V tile data at dst_index, fused sin/cos RoPE tensor.
 * Output: RoPE-transformed tile at dst_index.
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

inline void _rope_configure_addrmod_() {
    // SFPU configuration shifts to use ADDRMOD4-7.
    constexpr uint32_t ADDRMOD_OFFSET = 4;

    addr_mod_t{
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_0);

    addr_mod_t{
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_1);

    addr_mod_t{
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_2);

    addr_mod_t{
        .dest = {.incr = 0, .clr = 0, .cr = 0, .c_to_cr = 0},
    }
        .set(ADDRMOD_OFFSET + ADDR_MOD_3);
}

template <uint32_t row>
inline void _calculate_rope_row_group_() {
    //-------------------------------------------------------------------------
    // Step 1: Load RoPE tables from tile 2 (offset +128).
    //-------------------------------------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 128);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 2);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 4);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 6);

    //-------------------------------------------------------------------------
    // Step 2a: Load 4 token rows (even columns only, cos components)
    //-------------------------------------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, row);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 16);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 16);

    //-------------------------------------------------------------------------
    // Step 2b: Compute ac and store to scratch
    //-------------------------------------------------------------------------
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // LREG4 = a*c0
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // LREG5 = a*c1
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);  // LREG6 = a*c2
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);  // LREG7 = a*c3
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store the result (scratch 0)
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 16);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 18);

    //-------------------------------------------------------------------------
    // Step 3a: Load 4 token rows (even columns only, cos components)
    //-------------------------------------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, row);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 16);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 16);

    //-------------------------------------------------------------------------
    // Step 3b: Compute bc and store to scratch
    //-------------------------------------------------------------------------
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // LREG4 = b*c0
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // LREG5 = b*c1
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);  // LREG6 = b*c2
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);  // LREG7 = b*c3
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store the result (scratch 1)
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 4);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 4 + 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 4 + 16);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 4 + 18);

    //-------------------------------------------------------------------------
    // Step 4a: Load 4 token rows (odd columns only, sin components)
    //-------------------------------------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 2);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 16 + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 2);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 16 + 2);

    //-------------------------------------------------------------------------
    // Step 4b: Compute ad and store to scratch
    //-------------------------------------------------------------------------
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // LREG4 = a*d0
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // LREG5 = a*d1
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);  // LREG6 = a*d2
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);  // LREG7 = a*d3
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store the result (scratch 2)
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 8);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 8 + 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 8 + 16);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 8 + 18);

    //-------------------------------------------------------------------------
    // Step 5a: Load 4 token rows (odd columns only, sin components)
    //-------------------------------------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 2);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 16 + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 2);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 16 + 2);

    //-------------------------------------------------------------------------
    // Step 5b: Compute bd and store to scratch
    //-------------------------------------------------------------------------
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG4, 0);  // LREG4 = b*d0
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);  // LREG5 = b*d1
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);  // LREG6 = b*d2
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);  // LREG7 = b*d3
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store the result (scratch 3)
    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 12);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 12 + 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 12 + 16);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 12 + 18);

    //-------------------------------------------------------------------------
    // Step 6a: Load ac and bd from scratch 0 and scratch 3
    //-------------------------------------------------------------------------
    // Load ac from scratch 0
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 2);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 16);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 18);

    // Load bd from scratch 3
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 12);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 12 + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 12 + 16);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 12 + 18);

    //-------------------------------------------------------------------------
    // Step 6b: Compute ac - bd and store to scratch
    //-------------------------------------------------------------------------
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0);  // LREG0 = ac0 - bd0
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG1, 0);  // LREG1 = ac1 - bd1
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LCONST_neg1, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // LREG2 = ac2 - bd2
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LCONST_neg1, p_sfpu::LREG3, p_sfpu::LREG3, 0);  // LREG3 = ac3 - bd3
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store the result (original location of cos components)
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, row);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 16);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 16);

    //-------------------------------------------------------------------------
    // Step 7a: Load bc and ad from scratch 1 and scratch 2
    //-------------------------------------------------------------------------
    // Load bc from scratch 1
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 4);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 4 + 2);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 4 + 16);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 4 + 18);

    // Load ad from scratch 2
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 8);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 8 + 2);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 8 + 16);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 128 + 32 + 8 + 18);

    //-------------------------------------------------------------------------
    // Step 7b: Compute bc + ad and store to scratch
    //-------------------------------------------------------------------------
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG0, 0);  // LREG0 = bc0 + ad0
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG1, 0);  // LREG1 = bc1 + ad1
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // LREG2 = bc2 + ad2
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG3, 0);  // LREG3 = bc3 + ad3
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Store the result (original location of sin components)
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 2);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 16 + 2);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 2);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, row + 64 + 16 + 2);
}

inline void _calculate_rope_() {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    _calculate_rope_row_group_<0>();
    _calculate_rope_row_group_<4>();
    _calculate_rope_row_group_<8>();
    _calculate_rope_row_group_<12>();

    _calculate_rope_row_group_<32 + 0>();
    _calculate_rope_row_group_<32 + 4>();
    _calculate_rope_row_group_<32 + 8>();
    _calculate_rope_row_group_<32 + 12>();
}

}  // namespace sfpu

inline void _llk_math_rope_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(ckernel::sfpu::_rope_configure_addrmod_);
}

inline void _llk_math_rope_tile_(uint32_t dst_index) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_calculate_rope_, dst_index, VectorMode::RC_custom);
}

#endif

/**
 * @brief Initializes the RoPE SFPU operation.
 */
inline void rope_tile_init() { MATH((_llk_math_rope_tile_init_())); }

ALWI void rope_tile(uint32_t dst_index) { MATH((_llk_math_rope_tile_(dst_index))); }

}  // namespace ckernel
