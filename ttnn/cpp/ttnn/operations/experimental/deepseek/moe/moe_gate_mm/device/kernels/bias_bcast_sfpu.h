// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"

/**
 * @brief Broadcasts a 1-row bias and adds it element-wise to all 32 rows of a tile.
 *
 * Loads 4 FP16B bias values from the bias tile (DST offset 128, i.e. tile 2, face 0
 * rows 0-3) into LREG4-7. These 4 values, after transpose, represent a single bias
 * value broadcast across all 32 lanes. Then iterates over all 8 groups of 4 rows in
 * the input tile (4 groups in the upper faces, 4 in the lower faces), loading each
 * group, transposing, performing SFPMAD (input * 1.0 + bias), transposing back, and
 * storing in-place. The first group's instruction sequence is recorded into the replay
 * buffer and replayed for the remaining 3 groups per half-tile.
 *
 * Input:  32 FP16B values per lane from the DST tile at `input_index` (all 4 faces,
 *         32 rows total). 4 FP16B bias values from the DST tile at offset 128 (tile 2).
 * Output: 32 FP16B values per lane, written back in-place to the same DST tile at
 *         `input_index`. Each value is incremented by the broadcast bias.
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

inline void _add_bias_configure_addrmod_() {
    // SFPU configuration shifts to use ADDRMOD4-7.
    constexpr uint32_t ADDRMOD_OFFSET = 4;

    addr_mod_t{
        .dest = {.incr = -18, .clr = 0, .cr = 0, .c_to_cr = 0},
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

inline void _add_bias_() {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Let us load in the bias values
    TTI_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_1, 128);
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_2, 128);
    TTI_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_1, 128);
    TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 128);

    lltt::record<lltt::Exec>(0, 4 + 1 + 5 + 1 + 4);
    // Now load in the input, 4 rows at a time
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Now add the bias values to the input
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG3, 0);
    TTI_SFPNOP;

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Now store the output
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 0);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 0);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 0);

    // Now do this 3 more times, to complete first two faces (16 rows)
    for (uint32_t i = 0; i < 3; ++i) {
        lltt::replay(0, 15);
    }

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    lltt::record<lltt::Exec>(15, 4 + 1 + 5 + 1 + 4);
    // Now load in the input, 4 rows at a time
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 32 + 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 32 + 0);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 32 + 0);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, 32 + 0);

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Now add the bias values to the input
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG3, 0);
    TTI_NOP;

    TTI_SFPTRANSP(0, 0, 0, 0);

    // Now store the output
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_1, 32 + 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_2, 32 + 0);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_1, 32 + 0);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_3, 32 + 0);

    // Now do this 3 more times, to complete lower two faces (16 rows)
    for (uint32_t i = 0; i < 3; ++i) {
        lltt::replay(15, 15);
    }
}

}  // namespace sfpu

inline void _llk_math_add_bias_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(
        ckernel::sfpu::_add_bias_configure_addrmod_);
}

inline void _llk_math_add_bias_(uint32_t input_index) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_add_bias_, input_index, VectorMode::RC_custom);
}

#endif

/**
 * @brief Initializes the bias-broadcast-add SFPU operation.
 */
inline void add_bias_init() { MATH((_llk_math_add_bias_init_())); }

ALWI void add_bias(uint32_t input_index) { MATH((_llk_math_add_bias_(input_index))); }

}  // namespace ckernel
