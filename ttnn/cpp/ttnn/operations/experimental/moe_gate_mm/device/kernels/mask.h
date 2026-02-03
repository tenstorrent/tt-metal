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

inline void _mask_group_configure_addrmod_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_0);
}

template <uint32_t tile_index>
inline void _mask_group_() {
    // Extract bit for this tile from LREG4
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);  // Copy bitmask to LREG5

    // Shift right by tile_index to get the bit for this tile into LSB position
    // int shift_imm[8] = {0xFFF, 0xFFE, 0xFFD, 0xFFC, 0xFFB, 0xFFA, 0xFF9, 0xFF8};
    TTI_SFPSHFT((-tile_index) & 0xfff, 0, p_sfpu::LREG5, /*MOD=IMM1*/ 1);

    // Mask to get only bit 0
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_USHORT, 0x1);
    // TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, 0x0);
    TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG5, 0);  // LREG5 now has 0 or 1

    // Let us broadcast this value to all 32 bits of the lane
    for (uint32_t i = 0; i < 5; i++) {
        TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG6, 0);
        TTI_SFPSHFT(1 << i, 0, p_sfpu::LREG6, /*MOD=IMM1*/ 1);
        TTI_SFPOR(0, p_sfpu::LREG6, p_sfpu::LREG5, 0);
    }

    // Get the inversion of the mask
    TTI_SFPNOT(0, p_sfpu::LREG5, p_sfpu::LREG6, 0);

    // Record the batch processing sequence
    lltt::record<lltt::NoExec>(0, 16);

    for (uint32_t lreg = p_sfpu::LREG0; lreg < p_sfpu::LREG4; lreg++) {
        TTI_SFPAND(0, p_sfpu::LREG5, lreg, 0);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0xFF80);
        TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG7, 0);  // Select -inf if mask=0
        TTI_SFPOR(0, p_sfpu::LREG7, lreg, 0);            // Combine
    }

    // Now apply this for the entire tile
    uint32_t face_offset = 0;
    for (uint32_t face = 0; face < 4; face += 2) {
        uint32_t row_offset = face_offset;
        for (uint32_t row = 0; row < 16; row += 4) {
            // Step 1: Load from dst
            TTI_SFPTRANSP(0, 0, 0, 0);
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, row_offset + 0);
            TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, row_offset + 2);
            TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, row_offset + 16);
            TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, row_offset + 18);
            TTI_SFPTRANSP(0, 0, 0, 0);

            lltt::replay(0, 16);

            // Store back
            TTI_SFPTRANSP(0, 0, 0, 0);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, row_offset + 0);
            TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, row_offset + 2);
            TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, row_offset + 16);
            TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, row_offset + 18);
            TTI_SFPTRANSP(0, 0, 0, 0);

            row_offset += 4;
        }
        face_offset += 32;
    }
}

}  // namespace sfpu

inline void _llk_math_mask_group_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(
        ckernel::sfpu::_mask_group_configure_addrmod_);
}

template <uint32_t tile_index>
inline void _llk_math_mask_group_(uint32_t dst_index) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_mask_group_<tile_index>, dst_index, VectorMode::RC_custom);
}

#endif

/**
 * @brief Initializes the SFPU for mask operation
 * @return None. Modifies the first two rows of the first face of the tile with the result.
 */
inline void mask_group_init() { MATH((_llk_math_mask_group_init_())); }

/**
 * @brief Calculates the mask of the group
 * @return None. Modifies each tile in place.
 */
template <uint32_t tile_index>
ALWI void mask_group(uint32_t dst_index) {
    MATH((_llk_math_mask_group_<tile_index>(dst_index)));
}

}  // namespace ckernel
