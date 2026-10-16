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

/**
 * Bank 0 as _llk_pack_hw_configure_ leaves it for Float16_b, a 2048-byte 32x32 tile, four faces and no relu,
 * followed by _llk_pack_dest_init_. The strides it would program are written straight to their poisoned value.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_configure_current_bank_()
{
    constexpr std::uint32_t canonical_format = to_underlying(DataFormat::Float16_b);
    constexpr auto config                    = hw_cleanup::words_of(
        []
        {
            pack_config_t f {};
            f.exp_section_size = 4;
            f.uncompress       = 1;
            f.out_data_format  = canonical_format;
            f.in_data_format   = canonical_format;
            return f;
        }());
    constexpr auto dest_rd_ctrl = hw_cleanup::words_of(
        []
        {
            dest_rd_ctrl_t f {};
            f.PCK_DEST_RD_CTRL_Read_32b_data = is_fp32_dest_acc_en;
            return f;
        }());
    constexpr auto pack_counters = hw_cleanup::words_of(
        []
        {
            pack_counters_t f {};
            f.pack_reads_per_xy_plane = 1;
            return f;
        }());
    constexpr auto edge_offset = hw_cleanup::words_of(
        []
        {
            pck_edge_offset_t f {};
            f.mask = 0xffff;
            return f;
        }());
    volatile std::uint32_t tt_reg_ptr *cfg = hw_cleanup::bank0_cfg();

    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = 4 << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::TILE_HEADER]       = HW_CLEANUP_CANONICAL_TILE_SIZE_16B;
    regfile[p_gpr_pack::TILE_HEADER + 1]   = 0;
    regfile[p_gpr_pack::TILE_HEADER + 2]   = 0;
    regfile[p_gpr_pack::TILE_HEADER + 3]   = 0;
    regfile[p_gpr_pack::TMP0]              = 0;
    regfile[p_gpr_pack::TMP1]              = 0;

    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    hw_cleanup::cfg_rmw<THCON_SEC0_REG1_Pac_LF8_4b_exp_RMW, 0>();
    hw_cleanup::cfg_rmw<ALU_FORMAT_SPEC_REG2_Dstacc_RMW, canonical_format>();
    hw_cleanup::cfg_rmw<ALU_FORMAT_SPEC_REG_Dstacc_val_ADDR32, 0, ALU_FORMAT_SPEC_REG_Dstacc_val_MASK | ALU_FORMAT_SPEC_REG_Dstacc_override_MASK, 0>();
    hw_cleanup::cfg_rmw<STACC_RELU_ApplyRelu_ADDR32, 0, STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK, 0>();
    hw_cleanup::cfg_rmw<THCON_SEC0_REG1_Exp_threshold_ADDR32, 0, THCON_SEC0_REG1_Exp_threshold_en_MASK | THCON_SEC0_REG1_Exp_threshold_MASK, 0>();
    hw_cleanup::cfg_rmw<THCON_SEC0_REG1_Pack_L1_Acc_RMW, 0>();

    reset_dest_offset_id();
    TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO, p_cfg::WRCFG_32b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    TTI_DMANOP;
    TTI_DMANOP;
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;

    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32]             = dest_rd_ctrl.val[0];
    for (std::uint32_t i = 0; i < NUM_PACKERS; i++)
    {
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val[0];
    }
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = edge_offset.val[0];
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0;

    sync_regfile_write(p_gpr_pack::TMP1);
}

/**
 * Poison pack MOP / strides / PAC X so a following MicroOp cannot rely on
 * ambient Default pack state for those fields. Ops must call pack_init or
 * pack_reconfig_data_format<true> (or an equivalent full pack init) before packing.
 *
 * Default pack ADDR_MOD is restored: pack_reconfig_data_format<true> skips
 * addrmod config, and pack_block_contiguous_init only replaces the MOP.
 */
inline void _llk_pack_hw_cleanup_poison_pack_ambient_()
{
    _llk_pack_configure_addrmod_<PackMode::Default>();

    // 1x1 NOP double-loop so accidental MOP runs are inert.
    hw_cleanup::program_nop_mop();

    // Wrong PAC X (Default pack_init programs FACE_C_DIM - 1).
    TTI_SETADCXX(p_setadc::PAC, 0, 0x0);
}

/**
 * @brief Quiesces all TRISCs and restores pack cfg bank 0 to canonical Float16_b 32x32 geometry.
 * @tparam Dst Kernel DST_SYNC_MODE (compile-time; not modified by cleanup).
 * @tparam is_fp32_dest_acc_en Kernel DST_ACCUM_MODE (re-asserted, not changed).
 * @note On return cfg bank 0 uses Float16_b pack formats with 32x32 tiles and four faces, and
 *       is selected.
 * @note On return the pack-dest counters and Dest sync match startup and the Default pack
 *       ADDR_MOD is restored, while the pack MOP, strides and PAC X are deliberately poisoned.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_hw_cleanup_canonical_()
{
    hw_cleanup::start<PackThreadId>();

    hw_cleanup::select_cfg_state(0);
    _llk_pack_hw_cleanup_configure_current_bank_<Dst, is_fp32_dest_acc_en>();

    _llk_pack_hw_cleanup_poison_pack_ambient_();

    hw_cleanup::finish<PackThreadId>();
}
