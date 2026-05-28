// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
using namespace ckernel;

/**
 * @brief Configures MOP for binary unpack where one operand is tilized
 *
 * @tparam UNP_TILIZE_SEL: Which unpacker performs tilize,
 * values = p_unpacr::UNP_A / p_unpacr::UNP_B
 * @param buf_desc_id_0/1: Buffer descriptor ID for each operand
 */
template <std::uint32_t UNP_TILIZE_SEL>
inline void _llk_unpack_tilize_binary_operands_mop_config_(const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1)
{
    static_assert((UNP_TILIZE_SEL == p_unpacr::UNP_A) || (UNP_TILIZE_SEL == p_unpacr::UNP_B), "UNP_TILIZE_SEL can only be set to p_unpacr::UNP_A/UNP_B");

    const std::uint32_t MOP_OUTER_LOOP     = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    // For regular unpacking, use the unpacker that is not UNP_TILIZE_SEL
    std::uint32_t unpack_tile_instrn;
    if constexpr (UNP_TILIZE_SEL == p_unpacr::UNP_A)
    {
        unpack_tile_instrn = TT_OP_UNPACR1_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_1, 1 /*Set Dvalid*/);
    }
    else
    {
        unpack_tile_instrn = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    }

    std::uint32_t unpack_tilize_tile_instrn;
    const std::uint32_t tilize_buf_desc_id = UNP_TILIZE_SEL == p_unpacr::UNP_A ? buf_desc_id_0 : buf_desc_id_1;
    unpack_tilize_tile_instrn =
        TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z increment*/, 1 /*src Z increment*/, UNP_TILIZE_SEL, tilize_buf_desc_id, 1 /*Set Dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn, unpack_tilize_tile_instrn);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes binary unpack with tilize for one operand
 *
 * @tparam UNP_TILIZE_SEL: Which unpacker performs tilize,
 * values = p_unpacr::UNP_A / p_unpacr::UNP_B
 * @param buf_desc_id_0/1: Buffer descriptor ID for each operand
 * @param full_ct_dim: Number of tiles in a row of the input tensor which is to be tilized
 * @param tensor_shape: Contains all the information of the tile shape for the input which will be tilized: num faces, face row/col dim, etc
 */
template <std::uint32_t UNP_TILIZE_SEL>
inline void _llk_unpack_tilize_binary_operands_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t full_ct_dim, const TensorShape& tensor_shape)
{
    static_assert((UNP_TILIZE_SEL == p_unpacr::UNP_A) || (UNP_TILIZE_SEL == p_unpacr::UNP_B), "UNP_TILIZE_SEL can only be set to p_unpacr::UNP_A/UNP_B");

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);

    // Pack all UNPACK_TILIZE stride fields into a single struct to perform a direct 32-bit cfg write
    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.src_z_stride                       = tensor_shape.num_faces_c_dim; // col dim of a tile in L1 in units of 16 datums (1 face). This is used for
                                                                                  //  Src (L1) counter increments in the UNPACR_TILIZE instruction
    unpk_cfg.f.dst_z_stride      = 1;                                             // col dim of a tile in dest reg (1 face)
    unpk_cfg.f.stride_val_source = 0;
    unpk_cfg.f.stride_offset_0   = full_ct_dim * tensor_shape.num_faces_c_dim; // how much to stride to go to next row within the same tile

    if constexpr (UNP_TILIZE_SEL == p_unpacr::UNP_A)
    {
        cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }
    else
    {
        cfg[THCON_UNPACKER1_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
        cfg[THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];
    }

    _llk_unpack_tilize_binary_operands_mop_config_<UNP_TILIZE_SEL>(buf_desc_id_0, buf_desc_id_1);
}

/**
 * @brief Unpacks binary operands (to SrcA and SrcB), tilizing the operand selected by UNP_TILIZE_SEL
 *
 * @tparam UNP_TILIZE_SEL: Which unpacker performs tilize,
 * values = p_unpacr::UNP_A / p_unpacr::UNP_B
 * @param start_l1_tile_idx_0: L1 index for UNPACKER0, unpacks to SrcA
 * @param start_l1_tile_idx_1: L1 index for UNPACKER1, unpacks to SrcB
 */
template <std::uint32_t UNP_TILIZE_SEL>
inline void _llk_unpack_tilize_binary_operands_(const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1)
{
    static_assert((UNP_TILIZE_SEL == p_unpacr::UNP_A) || (UNP_TILIZE_SEL == p_unpacr::UNP_B), "UNP_TILIZE_SEL can only be set to p_unpacr::UNP_A/UNP_B");

    constexpr auto tilize_idx_mode  = p_set_inc_sel::FACE_SEL;
    constexpr auto regular_idx_mode = p_set_inc_sel::TILE_SEL;

    constexpr auto idx_mode_a = UNP_TILIZE_SEL == p_unpacr::UNP_A ? tilize_idx_mode : regular_idx_mode;
    constexpr auto idx_mode_b = UNP_TILIZE_SEL == p_unpacr::UNP_B ? tilize_idx_mode : regular_idx_mode;

    TT_SET_SRC_TILE_FACE_ROW_IDX(idx_mode_a, p_unpacr::UNP_A, start_l1_tile_idx_0);
    TT_SET_SRC_TILE_FACE_ROW_IDX(idx_mode_b, p_unpacr::UNP_B, start_l1_tile_idx_1);
    TTI_SET_DST_TILE_FACE_ROW_IDX(idx_mode_a, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(idx_mode_b, p_unpacr::UNP_B, 0);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
