// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(ARCH_BLACKHOLE)

constexpr std::uint32_t sdo_datum_bytes(std::uint32_t f) { return (f & 0x3) == 0 ? 4 : ((f & 0x3) == 1 ? 2 : 1); }

#ifdef TRISC_PACK
static_assert(
    PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT == 16 && PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT == 0 &&
    PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT == 16);

ALWI void sdo_gpr_word(std::uint32_t gpr, std::uint32_t value) {
    TT_SETDMAREG(0, value & 0xFFFF, 0, LO_16(gpr));
    TT_SETDMAREG(0, value >> 16, 0, HI_16(gpr));
}

ALWI void sdo_pack_strides(std::uint32_t src) {
    using namespace ckernel;
    const std::uint32_t row = FACE_C_DIM * sdo_datum_bytes(src);
    sdo_gpr_word(p_gpr_pack::TMP0, row << 16);
    sdo_gpr_word(p_gpr_pack::TMP1, (FACE_R_DIM * row) | ((4 * FACE_R_DIM * row) << 16));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
}

ALWI void sdo_pack_setup(std::uint32_t cb) {
    using namespace ckernel;
    TTI_SETC16(ADDR_MOD_PACK_SEC0_YsrcIncr_ADDR32, 0x0004);
    TTI_SETC16(ADDR_MOD_PACK_SEC1_YsrcIncr_ADDR32, 0x2020);
    TTI_SETC16(ADDR_MOD_PACK_SEC2_YsrcIncr_ADDR32, 0x1020);
    sdo_pack_strides(pack_src_format[cb]);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
}

ALWI void sdo_pack_clamp(bool zero) {
    using namespace ckernel;
    constexpr std::uint32_t field = STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    TTI_ATGETM(mutex::REG_RMW);
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, field>((zero ? 1u : 0u) << STACC_RELU_ApplyRelu_SHAMT);
    TTI_ATRELM(mutex::REG_RMW);
}
#endif

ALWI void sdo_pack_init(std::uint32_t cb) { PACK((sdo_pack_setup(cb))); }

ALWI void sdo_pack_relu(bool zero) { PACK((sdo_pack_clamp(zero))); }

#endif
