// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "experimental/llk_hw_cleanup.h"
#include "llk_unpack_common.h"

using namespace ckernel;

/**
 * Bank 0 as _llk_unpack_hw_configure_ leaves it for Float16_b operands A and B with 16-row faces, four faces and
 * 2048-byte tiles.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_hw_cleanup_configure_current_bank_()
{
    constexpr std::uint32_t canonical_format = to_underlying(DataFormat::Float16_b);
    constexpr std::uint32_t face_dim         = FACE_R_DIM * FACE_C_DIM;
    constexpr std::uint32_t z_stride         = canonical_unpA_z_stride(canonical_format);
    constexpr std::uint32_t alu_mask         = ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK | ALU_FORMAT_SPEC_REG0_SrcBUnsigned_MASK |
                                       ALU_ROUNDING_MODE_Fpu_srnd_en_MASK | ALU_ROUNDING_MODE_Gasket_srnd_en_MASK | ALU_ROUNDING_MODE_Packer_srnd_en_MASK;
    constexpr auto tile_descriptor = [](const std::uint32_t x_dim)
    {
        unpack_tile_descriptor_t f {};
        f.in_data_format = canonical_format;
        f.uncompressed   = 1;
        f.x_dim          = x_dim;
        f.y_dim          = 1;
        f.z_dim          = 4;
        return hw_cleanup::words_of(f);
    };
    constexpr auto descriptor_a = tile_descriptor(0);
    constexpr auto descriptor_b = tile_descriptor(face_dim);
    constexpr auto config       = hw_cleanup::words_of(
        []
        {
            unpack_config_t f {};
            f.out_data_format    = canonical_format;
            f.throttle_mode      = 2;
            f.uncompress_cntx0_3 = 0xf;
            f.uncompress_cntx4_7 = 0xf;
            return f;
        }());
    volatile std::uint32_t tt_reg_ptr *cfg = hw_cleanup::bank0_cfg();

    regfile[p_gpr_unpack::FACE_DIM_16x16] = (face_dim / 1) | ((face_dim / 1) << 16);
    regfile[p_gpr_unpack::FACE_DIM_8x16]  = (face_dim / 2) | ((face_dim / 2) << 16);
    regfile[p_gpr_unpack::FACE_DIM_4x16]  = (face_dim / 4) | ((face_dim / 4) << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16]  = (face_dim / 8) | ((face_dim / 8) << 16);
    regfile[p_gpr_unpack::FACE_DIM_1x16]  = (face_dim / 16) | ((face_dim / 16) << 16);

    unpacker_addr_counter_init();
    hw_cleanup::cfg_rmw<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK,
        canonical_unpA_y_stride(canonical_format)>();
    hw_cleanup::cfg_rmw<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_mask, 0>();
    hw_cleanup::cfg_rmw<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW, 0>();
    hw_cleanup::cfg_rmw<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW, 0>();
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);
    reset_config_context();
    TTI_SETDMAREG(0, HW_CLEANUP_CANONICAL_TILE_SIZE_16B, 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    TTI_SETDMAREG(0, HW_CLEANUP_CANONICAL_TILE_SIZE_16B, 0, LO_16(p_gpr_unpack::TILE_SIZE_B));

    cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
    cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = z_stride << UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = descriptor_a.val[i];
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = descriptor_b.val[i];
    }
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = config.val[i];
        cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    }
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32]         = (4 * 16) | ((4 * 16) << 16);
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32]           = face_dim | (face_dim << 16);
    cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;

    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);
}

/**
 * @brief Quiesces all TRISCs and restores unpack cfg bank 0 to canonical Float16_b 32x32 geometry.
 * @tparam is_fp32_dest_acc_en Kernel DST_ACCUM_MODE (re-asserted, not changed).
 * @note On return cfg bank 0 uses Float16_b 32x32 tiles with four faces (2048 bytes) and is
 *       selected.
 * @note On return MOP CFG is NOP-poisoned to a 1x1 double-loop template.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_hw_cleanup_canonical_()
{
    hw_cleanup::start<UnpackThreadId>();

    hw_cleanup::select_cfg_state(0);
    _llk_unpack_hw_cleanup_configure_current_bank_<is_fp32_dest_acc_en>();

    // 1x1 NOP double-loop so accidental MOP runs (template or unpack-template) are inert.
    hw_cleanup::program_nop_mop();

    hw_cleanup::finish<UnpackThreadId>();
}
