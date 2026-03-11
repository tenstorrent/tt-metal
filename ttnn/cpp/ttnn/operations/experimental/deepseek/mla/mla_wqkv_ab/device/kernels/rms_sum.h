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

inline void _rms_sum_configure_addrmod_() {
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

inline void _rms_sum_calculate_tile_() {
    // Reset Dst RWC to 0
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // Running accumulator for sum(x^2)
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);

    //-------------------------------------------------------------------------
    // q_norm: First 9 cores
    //-------------------------------------------------------------------------
    lltt::record<lltt::Exec>(0, 8);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 2);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 4);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_2, 6);

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Now do the same for data from 8 other cores (total 9 cores)
    for (uint32_t core_id = 1; core_id < 9; ++core_id) {
        lltt::replay(0, 8);
    }

    //-------------------------------------------------------------------------
    // k_norm: Last 3 cores
    //-------------------------------------------------------------------------
    lltt::record<lltt::Exec>(8, 8);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 2);
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 4);
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_2, 6);

    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG5, 0);
    TTI_SFPNOP;
    TTI_SFPTRANSP(0, 0, 0, 0);

    // Now do the same for data from 2 other cores (total 3 cores)
    lltt::replay(8, 6);
    lltt::replay(8, 6);

    // Write output to destination tile
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);

    TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);
    TTI_SFPTRANSP(0, 0, 0, 0);

    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP16B, ADDR_MOD_0, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP16B, ADDR_MOD_0, 2);
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP16B, ADDR_MOD_0, 16);
    TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::FP16B, ADDR_MOD_0, 18);

    TTI_SFPSTORE(p_sfpu::LREG4, InstrModLoadStore::FP16B, ADDR_MOD_0, 64 + 0);
    TTI_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::FP16B, ADDR_MOD_0, 64 + 2);
    TTI_SFPSTORE(p_sfpu::LREG6, InstrModLoadStore::FP16B, ADDR_MOD_0, 64 + 16);
    TTI_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::FP16B, ADDR_MOD_0, 64 + 18);
}

}  // namespace sfpu

inline void _llk_math_rms_sum_tile_init_() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, /*APPROXIMATE=*/true>(
        ckernel::sfpu::_rms_sum_configure_addrmod_);
}

inline void _llk_math_rms_sum_single_tile_(uint32_t dst_index) {
    _llk_math_eltwise_unary_sfpu_params_</*APPROXIMATE=*/true>(
        ckernel::sfpu::_rms_sum_calculate_tile_, dst_index, VectorMode::RC_custom);
}

#endif

/**
 * @brief Initializes the RMS sum SFPU operation.
 */
inline void rms_sum_init() { MATH((_llk_math_rms_sum_tile_init_())); }

/**
 * @brief Sum(x^2) reduction for two separate RMS sums.
 */
ALWI void rms_sum_tile(uint32_t dst_index) { MATH((_llk_math_rms_sum_single_tile_(dst_index))); }

}  // namespace ckernel
