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
 * @brief Configures MOP for binary unpack where one operand (-> SRCA) is tilized, compatible with math reduce operations.
 *
 * Tilizes a full face into SrcA. This MOP is called multiple times to tilize a full tile in the execute function.
 *
 * @param buf_desc_id_0: Buffer descriptor ID for the operand that will be tilized
 * @param full_ct_dim: Number of tiles in a row of the input tensor
 * @param tensor_shape: Contains all the information of the tile shape for the input that will be tilized: num faces, face row/col dim, etc
 */
inline void _llk_unpack_tilize_operands_reduce_mop_config_(const std::uint32_t buf_desc_id_0, const std::uint32_t full_ct_dim, const TensorShape& tensor_shape)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    const std::uint32_t unpack_half_face_instrn =
        TT_OP_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 0 /*Set Dvalid*/);
    const std::uint32_t unpack_half_face_instrn_dvalid = TT_OP_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);

    const std::uint32_t increment_half_face_instrn = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(
        p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, tensor_shape.num_faces_c_dim * ckernel::unpack::UNPACR_STRIDE_MAX_ROWS * full_ct_dim);

    // Unpack first half face, increment to second half face, unpack second half face
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_half_face_instrn, increment_half_face_instrn);
    temp.set_end_op(unpack_half_face_instrn_dvalid);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes unpacker to tilize a full tile for SrcA and unpack a single face for SrcB (scalar), compatible with math reduce kernels.
 *
 * @param buf_desc_id_0: Buffer descriptor ID for the operand that will be tilized
 * @param full_ct_dim: Number of tiles in a row of the input tensor
 * @param tensor_shape: Contains all the information of the tile shape for the input that will be tilized: num faces, face row/col dim, etc
 */
inline void _llk_unpack_tilize_operands_reduce_init_(const std::uint32_t buf_desc_id_0, const std::uint32_t full_ct_dim, const TensorShape& tensor_shape)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);

    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.stride_val_source                  = 0;
    unpk_cfg.f.stride_offset_0                    = full_ct_dim * tensor_shape.num_faces_c_dim; // stride to next row within same tile

    cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
    cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];

    _llk_unpack_tilize_operands_reduce_mop_config_(buf_desc_id_0, full_ct_dim, tensor_shape);
}

/**
 * @brief Unpacks and tilizes a full tile for SrcA and unpacks a single face for SrcB (scalar), compatible with math reduce kernels.
 *
 * Unpacks a single face for SrcB (scalar) and tilizes a full tile into SrcA. InputA tile is tilized face by face, and the
 * L1 index is set to each face in the tile. The L1 index is in units derived from the x/y/z dimensions of the
 * buffer descriptor. The buffer descriptor should be set to x_dim = 16, y_dim = 1, z_dim = 1, so the unit is 1 row.
 *
 * @param buf_desc_id_1: Buffer descriptor ID for scaler operand
 * @param full_ct_dim: Number of tiles in a row of the input tensor
 * @param tensor_shape: Contains all the information of the tile shape for the input that will be tilized: num faces, face row/col dim, etc
 * @param start_l1_tile_idx_0: L1 index for UNPACKER0, unpacks to SrcA
 * @param start_l1_tile_idx_1: L1 index for UNPACKER1, unpacks to SrcB
 */
inline void _llk_unpack_tilize_operands_reduce_(
    const std::uint32_t buf_desc_id_1,
    const std::uint32_t full_ct_dim,
    const TensorShape& tensor_shape,
    const std::uint32_t start_l1_tile_idx_0,
    const std::uint32_t start_l1_tile_idx_1)
{
    // Set L1 and src reg indices for unpacking the scaler input face into SrcB
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, start_l1_tile_idx_1);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // Unpack scaler input face into SrcB
    TT_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);

    // Set L1 and src reg indices for tilizing F0 into SrcA
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, start_l1_tile_idx_0 * tensor_shape.num_faces_c_dim);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer); // Face 0

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, start_l1_tile_idx_0 * tensor_shape.num_faces_c_dim + 1); // set L1 index to F1
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);                                                                    // Face 1

    TT_SET_SRC_TILE_FACE_ROW_IDX(
        p_set_inc_sel::TILE_SEL,
        p_unpacr::UNP_A,
        start_l1_tile_idx_0 * tensor_shape.num_faces_c_dim + tensor_shape.num_faces_c_dim * tensor_shape.face_r_dim * full_ct_dim); // set L1 index to F2
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);                                                                    // Face 2

    TT_SET_SRC_TILE_FACE_ROW_IDX(
        p_set_inc_sel::TILE_SEL,
        p_unpacr::UNP_A,
        start_l1_tile_idx_0 * tensor_shape.num_faces_c_dim + tensor_shape.num_faces_c_dim * tensor_shape.face_r_dim * full_ct_dim + 1); // set L1 index to F3
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);                                                                        // Face 3
}
