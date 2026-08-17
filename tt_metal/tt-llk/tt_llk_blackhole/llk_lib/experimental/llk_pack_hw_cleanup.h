// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "experimental/llk_hw_cleanup.h"
#include "llk_defs.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

template <bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_configure_current_bank_()
{
    constexpr std::uint32_t canonical_format = to_underlying(DataFormat::Float16_b);
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        canonical_format, canonical_format, HW_CLEANUP_CANONICAL_TILE_SIZE_16B, FACE_R_DIM, TILE_C_DIM, 4, false, 0);
}

/**
 * Poison pack MOP / strides / PAC X so a following MicroOp cannot rely on
 * ambient Default pack state for those fields. Ops must call pack_init or
 * pack_reconfig_data_format<true> (or an equivalent full pack init) before packing.
 *
 * Default pack ADDR_MOD is restored: pack_reconfig_data_format<true> skips
 * addrmod config, and pack_block_contiguous_init only replaces the MOP.
 *
 * hw_configure above programs correct Default strides; overwrite them here.
 */
inline void _llk_pack_hw_cleanup_poison_pack_ambient_()
{
    _llk_pack_configure_addrmod_<PackMode::Default>();

    // 1x1 NOP double-loop so accidental MOP runs are inert.
    ckernel_template(1, 1).program();

    // Zero packer strides (X unused; Y/Z/W cleared).
    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;

    // Wrong PAC X (Default pack_init programs FACE_C_DIM - 1).
    TTI_SETADCXX(p_setadc::PAC, 0, 0x0);
}

/**
 * @brief Quiesces all TRISCs and restores pack cfg banks to canonical Float16_b 32x32 geometry.
 * @tparam Dst Kernel DST_SYNC_MODE (compile-time; not modified by cleanup).
 * @tparam is_fp32_dest_acc_en Kernel DST_ACCUM_MODE (re-asserted, not changed).
 * @note On return both cfg banks use Float16_b pack formats with 32x32 tiles and four faces,
 *       and bank 0 is selected.
 * @note On return the pack-dest counters and Dest sync match startup and the Default pack
 *       ADDR_MOD is restored, while the pack MOP, strides and PAC X are deliberately poisoned.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_canonical_()
{
    hw_cleanup::start<PackThreadId>();

    hw_cleanup::select_cfg_state(0);
    _llk_pack_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    hw_cleanup::select_cfg_state(1);
    _llk_pack_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    hw_cleanup::select_cfg_state(0);

    _llk_pack_dest_init_<Dst, is_fp32_dest_acc_en>();

    _llk_pack_hw_cleanup_poison_pack_ambient_();

    hw_cleanup::finish<PackThreadId>();
}
