// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#ifdef TRISC_MATH
ALWI void sdo_sfpu_frame(const bool row_pairs) {
    TTI_SFPCONFIG(0, 0xF, 1);
    TTI_SETC16(ADDR_MOD_AB_SEC0_SrcAIncr_ADDR32 + ADDR_MOD_7, 0);
    TTI_SETC16(ADDR_MOD_DST_SEC0_DestIncr_ADDR32 + ADDR_MOD_7, 0);
    TTI_SETC16(ADDR_MOD_BIAS_SEC0_BiasIncr_ADDR32 + ADDR_MOD_7, 0);
    if (row_pairs) {
        TTI_SETC16(ADDR_MOD_AB_SEC0_SrcAIncr_ADDR32 + ADDR_MOD_6, 0);
        TTI_SETC16(ADDR_MOD_DST_SEC0_DestIncr_ADDR32 + ADDR_MOD_6, 2);
        TTI_SETC16(ADDR_MOD_BIAS_SEC0_BiasIncr_ADDR32 + ADDR_MOD_6, 0);
    }
}
#endif

ALWI void sdo_sfpu_init_unary() {
    MATH((sdo_sfpu_frame(false)));
    MATH((TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F)));
}

template <bool row_pairs = false, class F>
ALWI void sdo_sfpu_init_unary(F&& init) {
    MATH((sdo_sfpu_frame(row_pairs)));
    MATH((TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F)));
    MATH((std::forward<F>(init)()));
}

template <class F>
ALWI void sdo_sfpu_init_binary(F&& init) {
    MATH((sdo_sfpu_frame(true)));
    MATH((TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F)));
    MATH((std::forward<F>(init)()));
}

ALWI void sdo_sfpu_start(const std::uint32_t idst) {
    MATH((TT_SETC16(
        DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (idst << 6) + (dest_offset_id ? DEST_REGISTER_HALF_SIZE : 0))));
    MATH((TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH)));
}

ALWI void sdo_sfpu_done() { MATH((TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D))); }

ALWI void sdo_sfpu_next_face() {
    MATH((TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)));
    MATH((TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)));
}
