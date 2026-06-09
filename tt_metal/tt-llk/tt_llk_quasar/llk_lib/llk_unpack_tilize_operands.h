// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

/**
 * @brief Configures MOP for binary unpack where one or two operands are tilized
 *
 * @tparam TILIZE_UNP_SEL: Which unpacker performs tilize,
 * values = TilizeUnpackerSel::UnpA / TilizeUnpackerSel::UnpB / TilizeUnpackerSel::UnpAB
 * @param buf_desc_id_0/1: Buffer descriptor ID for each operand
 */
template <TilizeUnpackerSel TILIZE_UNP_SEL>
inline void _llk_unpack_tilize_operands_mop_config_(const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1)
{
    static_assert(
        (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpA) || (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpB) || (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpAB),
        "TILIZE_UNP_SEL can only be set to TilizeUnpackerSel::UnpA/UnpB/UnpAB");

    const std::uint32_t MOP_OUTER_LOOP     = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    std::uint32_t unpack_instrn_0;
    std::uint32_t unpack_instrn_1;

    if constexpr (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpA)
    {
        unpack_instrn_0 = TT_OP_UNPACR1_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_1, 1 /*Set Dvalid*/);
        unpack_instrn_1 =
            TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z increment*/, 1 /*src Z increment*/, p_unpacr::UNP_A, buf_desc_id_0, 1 /*Set Dvalid*/);
    }
    else if constexpr (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpB)
    {
        unpack_instrn_0 = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
        unpack_instrn_1 =
            TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z increment*/, 1 /*src Z increment*/, p_unpacr::UNP_B, buf_desc_id_1, 1 /*Set Dvalid*/);
    }
    else // UnpAB
    {
        unpack_instrn_0 =
            TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z increment*/, 1 /*src Z increment*/, p_unpacr::UNP_A, buf_desc_id_0, 1 /*Set Dvalid*/);
        unpack_instrn_1 =
            TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z increment*/, 1 /*src Z increment*/, p_unpacr::UNP_B, buf_desc_id_1, 1 /*Set Dvalid*/);
    }

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn_0, unpack_instrn_1);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes binary unpack with tilize for one or two operands
 *
 * @tparam TILIZE_UNP_SEL: Which unpacker performs tilize,
 * values = TilizeUnpackerSel::UnpA / TilizeUnpackerSel::UnpB / TilizeUnpackerSel::UnpAB
 * @param buf_desc_id_0/1: Buffer descriptor ID for each operand
 * @param full_ct_dim: Number of tiles in a row of the input tensor
 * @param tensor_shape: Contains all the information of the tile shape for the input: num faces, face row/col dim, etc
 */
template <TilizeUnpackerSel TILIZE_UNP_SEL>
inline void _llk_unpack_tilize_operands_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t full_ct_dim, const TensorShape& tensor_shape)
{
    static_assert(
        (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpA) || (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpB) || (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpAB),
        "TILIZE_UNP_SEL can only be set to TilizeUnpackerSel::UnpA/UnpB/UnpAB");

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);

    // Pack all UNPACK_TILIZE stride fields into a single struct to perform a direct 32-bit cfg write
    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.src_z_stride                       = tensor_shape.num_faces_c_dim; // col dim of a tile in L1 in units of 16 datums (1 face). This is used for
                                                                                  //  Src (L1) counter increments in the UNPACR_TILIZE instruction
    unpk_cfg.f.dst_z_stride      = 1;                                             // col dim of a tile in dest reg (1 face)
    unpk_cfg.f.stride_val_source = 0;
    unpk_cfg.f.stride_offset_0   = full_ct_dim * tensor_shape.num_faces_c_dim; // how much to stride to go to next row within the same tile

    if constexpr (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpA)
    {
        cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }
    else if constexpr (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpB)
    {
        cfg[THCON_UNPACKER1_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }
    else // UnpAB
    {
        cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
        cfg[THCON_UNPACKER1_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }

    _llk_unpack_tilize_operands_mop_config_<TILIZE_UNP_SEL>(buf_desc_id_0, buf_desc_id_1);
}

/**
 * @brief Unpacks binary operands (to SrcA and SrcB), tilizing the operand(s) selected by TILIZE_UNP_SEL
 *
 * @tparam TILIZE_UNP_SEL: Which unpacker(s) perform tilize,
 * values = TilizeUnpackerSel::UnpA / TilizeUnpackerSel::UnpB / TilizeUnpackerSel::UnpAB
 * @param start_l1_tile_idx_0: L1 index for UNPACKER0, unpacks to SrcA
 * @param start_l1_tile_idx_1: L1 index for UNPACKER1, unpacks to SrcB
 */
template <TilizeUnpackerSel TILIZE_UNP_SEL>
inline void _llk_unpack_tilize_operands_(const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1)
{
    static_assert(
        (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpA) || (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpB) || (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpAB),
        "TILIZE_UNP_SEL can only be set to TilizeUnpackerSel::UnpA/UnpB/UnpAB");

    constexpr auto tilize_idx_mode  = p_set_inc_sel::FACE_SEL;
    constexpr auto regular_idx_mode = p_set_inc_sel::TILE_SEL;

    constexpr bool tilize_a = (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpA) || (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpAB);
    constexpr bool tilize_b = (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpB) || (TILIZE_UNP_SEL == TilizeUnpackerSel::UnpAB);

    constexpr auto idx_mode_a = tilize_a ? tilize_idx_mode : regular_idx_mode;
    constexpr auto idx_mode_b = tilize_b ? tilize_idx_mode : regular_idx_mode;

    TT_SET_SRC_TILE_FACE_ROW_IDX(idx_mode_a, p_unpacr::UNP_A, start_l1_tile_idx_0);
    TT_SET_SRC_TILE_FACE_ROW_IDX(idx_mode_b, p_unpacr::UNP_B, start_l1_tile_idx_1);
    TTI_SET_DST_TILE_FACE_ROW_IDX(idx_mode_a, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(idx_mode_b, p_unpacr::UNP_B, 0);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
