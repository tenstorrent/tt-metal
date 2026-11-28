// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "sd_own_pack.hpp"

#if defined(ARCH_BLACKHOLE)

constexpr bool sdo_known_format(std::uint32_t f) {
    return f == static_cast<std::uint32_t>(DataFormat::Float16_b) ||
           f == static_cast<std::uint32_t>(DataFormat::Bfp8_b) || f == static_cast<std::uint32_t>(DataFormat::Int32) ||
           f == static_cast<std::uint32_t>(DataFormat::Invalid);
}

template <typename T, std::size_t N>
constexpr bool sdo_formats_known(const T (&f)[N]) {
    for (std::size_t i = 0; i < N; i++) {
        if (!sdo_known_format(f[i])) {
            return false;
        }
    }
    return true;
}

template <typename T, std::size_t N>
constexpr bool sdo_all_equal(const T (&v)[N], std::uint32_t x) {
    for (std::size_t i = 0; i < N; i++) {
        if (v[i] != x) {
            return false;
        }
    }
    return true;
}

#ifdef TRISC_PACK
static_assert(sdo_formats_known(pack_src_format) && sdo_formats_known(pack_dst_format));
static_assert(sdo_all_equal(pack_tile_num_faces, 4) && sdo_all_equal(pack_tile_face_r_dim, 16));
static_assert(sdo_all_equal(pack_tile_c_dim, 32));
#else
static_assert(sdo_formats_known(unpack_src_format) && sdo_formats_known(unpack_dst_format));
static_assert(sdo_all_equal(unpack_tile_num_faces, 4) && sdo_all_equal(unpack_tile_face_r_dim, 16));
#endif

#ifdef TRISC_UNPACK
ALWI void sdo_unpack_fmt_a(std::uint32_t cb) {
    using namespace ckernel;
    const std::uint32_t src = unpack_src_format[cb];
    const std::uint32_t dst = unpack_dst_format[cb];
    const std::uint32_t row = FACE_R_DIM * sdo_datum_bytes(dst);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(src);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(dst);
    TT_SETDMAREG(0, LOWER_HALFWORD(get_local_cb_interface(cb).fifo_page_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(FACE_C_DIM * row);
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(row);
}

ALWI void sdo_unpack_fmt_b(std::uint32_t cb) {
    using namespace ckernel;
    const std::uint32_t src = unpack_src_format[cb];
    const std::uint32_t dst = unpack_dst_format[cb];
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(src);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(dst);
    TT_SETDMAREG(0, LOWER_HALFWORD(get_local_cb_interface(cb).fifo_page_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(FACE_C_DIM * FACE_R_DIM * sdo_datum_bytes(dst));
}

ALWI bool sdo_unpack_differs(std::uint32_t a, std::uint32_t b) {
    return unpack_src_format[a] != unpack_src_format[b] || unpack_dst_format[a] != unpack_dst_format[b];
}

ALWI void sdo_unpack_fmt_a_if(std::uint32_t old_cb, std::uint32_t new_cb) {
    if (sdo_unpack_differs(old_cb, new_cb)) {
        sdo_unpack_fmt_a(new_cb);
    }
}

ALWI void sdo_unpack_fmt_b_if(std::uint32_t old_cb, std::uint32_t new_cb) {
    if (sdo_unpack_differs(old_cb, new_cb)) {
        sdo_unpack_fmt_b(new_cb);
    }
}
#endif

#ifdef TRISC_MATH
ALWI void sdo_math_operands() {
    using namespace ckernel;
    using namespace ckernel::math;
    if (src_zero_flag_hw != 0) {
        src_zero_flag_hw = 0;
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
    }
}

ALWI void sdo_math_operands_if(std::uint32_t old_a, std::uint32_t new_a, std::uint32_t old_b, std::uint32_t new_b) {
    if (unpack_dst_format[old_a] != unpack_dst_format[new_a] || unpack_dst_format[old_b] != unpack_dst_format[new_b]) {
        sdo_math_operands();
    }
}
#endif

#ifdef TRISC_PACK
ALWI void sdo_pack_fmt(std::uint32_t cb) {
    using namespace ckernel;
    const std::uint32_t src = pack_src_format[cb];
    const std::uint32_t dst = pack_dst_format[cb];
    TT_SETDMAREG(0, 1 | (dst << 4) | (src << 8), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2);
    TT_SETDMAREG(0, LOWER_HALFWORD(get_local_cb_interface(cb).fifo_page_size), 0, LO_16(p_gpr_pack::TILE_HEADER));
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(src);
    sdo_pack_strides(src);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
}

ALWI void sdo_pack_fmt_if(std::uint32_t old_cb, std::uint32_t new_cb) {
    const std::uint32_t a = pack_dst_format[old_cb];
    const std::uint32_t b = pack_dst_format[new_cb];
    constexpr std::uint32_t none = static_cast<std::uint32_t>(DataFormat::Invalid);
    if (a != b && a != none && b != none) {
        sdo_pack_fmt(new_cb);
    }
}

ALWI void sdo_pack_accumulate(std::uint32_t enable) {
    using namespace ckernel;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(enable);
    cfg_reg_rmw_tensix<PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain_RMW>(enable ? 1 : 0);
}
#endif

ALWI void sdo_reconfig(std::uint32_t srca_cb, std::uint32_t srcb_cb) {
    UNPACK((sdo_unpack_fmt_a(srca_cb)));
    UNPACK((sdo_unpack_fmt_b(srcb_cb)));
    MATH((sdo_math_operands()));
}

ALWI void sdo_reconfig(std::uint32_t old_a, std::uint32_t new_a, std::uint32_t old_b, std::uint32_t new_b) {
    UNPACK((sdo_unpack_fmt_a_if(old_a, new_a)));
    UNPACK((sdo_unpack_fmt_b_if(old_b, new_b)));
    MATH((sdo_math_operands_if(old_a, new_a, old_b, new_b)));
}

ALWI void sdo_reconfig_srca(std::uint32_t cb) {
    UNPACK((sdo_unpack_fmt_a(cb)));
    MATH((sdo_math_operands()));
}

ALWI void sdo_reconfig_srca(std::uint32_t old_cb, std::uint32_t new_cb) {
    UNPACK((sdo_unpack_fmt_a_if(old_cb, new_cb)));
    MATH((sdo_math_operands_if(old_cb, new_cb, old_cb, old_cb)));
}

ALWI void sdo_pack_reconfig(std::uint32_t cb) { PACK((sdo_pack_fmt(cb))); }

ALWI void sdo_pack_reconfig(std::uint32_t old_cb, std::uint32_t new_cb) { PACK((sdo_pack_fmt_if(old_cb, new_cb))); }

ALWI void sdo_pack_l1_acc(std::uint32_t enable) { PACK((sdo_pack_accumulate(enable))); }

#endif
