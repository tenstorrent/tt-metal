// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

/**
 * @brief Configures MOP for binary unpack where one operand (-> SRCA) is tilized, compatible with math reduce column operations.
 *
 * Unpacks a single face for SrcB (scalar) and tilizes a full tile into SrcA. InputA tile is tilized face by face, and the
 * L1 index is set to each face in the tile. The L1 index is in units derived from the x/y/z dimensions of the
 * buffer descriptor. The buffer descriptor should be set to x_dim = 16, y_dim = 1, z_dim = 1, so the unit is 1 row.
 *
 * @param buf_desc_id_0: Buffer descriptor ID for the operand that will be tilized into srcA
 * @param buf_desc_id_1: Buffer descriptor ID for scaler operand that will be unpacked into srcB
 * @param full_ct_dim: Number of tiles in a row of the input tensor
 * @param tensor_shape: Contains all the information of the tile shape for the input that will be tilized: num faces, face row/col dim, etc
 */
inline void _llk_unpack_reduce_col_tilizeA_strided_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t full_ct_dim, const TensorShape& tensor_shape)
{
    const std::uint32_t idx_inc = ckernel::unpack::UNPACR_STRIDE_MAX_ROWS * tensor_shape.num_faces_c_dim * full_ct_dim;

    constexpr std::uint32_t kIndex18BitMask = 0x3FFFF; // index field is 18 bits wide
    // 18-bit two's complement of (idx_inc - 1)
    const std::uint32_t idx_dec = (1u - idx_inc) & kIndex18BitMask;

    constexpr std::uint32_t replay_buf_len = 15;

    load_replay_buf<0, replay_buf_len>(
        [buf_desc_id_0, idx_inc, idx_dec]
        {
            // Face 0
            TT_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 0 /*Set Dvalid*/);
            TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, idx_inc);
            TT_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);

            // Face 1
            TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, idx_dec);
            TT_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 0 /*Set Dvalid*/);
            TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, idx_inc);
            TT_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);

            // Face 2
            TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, idx_inc - 1);
            TT_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 0 /*Set Dvalid*/);
            TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, idx_inc);
            TT_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);

            // Face 3
            TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, idx_dec);
            TT_UNPACR0_STRIDE(ckernel::unpack::UNPACR_STRIDE_MAX_ROWS /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 0 /*Set Dvalid*/);
            TT_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, idx_inc);
            TT_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0, 0, 0, 0, buf_desc_id_0, 1 /*Set Dvalid*/);
        });

    std::uint32_t unpack_srcB_face = TT_OP_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    std::uint32_t unpack_srcA      = TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0);

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_srcA);
    temp.set_start_op(unpack_srcB_face);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Configures MOP for binary unpack of tiny tiles (Nx32 with N <= 8), where one operand (-> SRCA) is tilized,
 * compatible with math reduce column operations.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @param buf_desc_id_0: Buffer descriptor ID for the operand that will be tilized into srcA
 * @param buf_desc_id_1: Buffer descriptor ID for scaler operand that will be unpacked into srcB
 */
template <PoolType POOL_TYPE>
inline void _llk_unpack_reduce_col_tilizeA_strided_tiny_tiles_mop_config_(const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;
    constexpr std::uint32_t replay_buf_len = 4;

    constexpr std::uint32_t clr_mode = (POOL_TYPE == PoolType::MAX) ? p_unpacr::UNP_CLRSRC_NEGINF : p_unpacr::UNP_CLRSRC_ZERO;

    load_replay_buf<0, replay_buf_len>(
        [buf_desc_id_0]
        {
            // Face 0; Src_Reg_Y_Cntr_Incr = 0 resets the counter, so each face starts at row 0 of its own SrcA bank
            TTI_UNPACR_NOP(p_unpacr::UNP_A, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear curr bank */, clr_mode, p_unpacr::UNP_CLRSRC_ZERO /* UNP_CLR_SRC */);
            TT_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0 /*inc by 1*/, 1 /*set to inc*/, 1 /*Row_Mask_Reg_Sel*/, 0, buf_desc_id_0, 1 /*Set Dvalid*/);

            // Face 1
            TTI_UNPACR_NOP(p_unpacr::UNP_A, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear curr bank */, clr_mode, p_unpacr::UNP_CLRSRC_ZERO /* UNP_CLR_SRC */);
            TT_UNPACR0_STRIDE(0 /*Src_Reg_Y_Cntr_Incr*/, 0 /*inc by 1*/, 1 /*set to inc*/, 1 /*Row_Mask_Reg_Sel*/, 0, buf_desc_id_0, 1 /*Set Dvalid*/);
        });

    std::uint32_t unpack_srcB_face = TT_OP_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    std::uint32_t unpack_srcA      = TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_srcA);
    temp.set_start_op(unpack_srcB_face);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes unpacker to tilize a tile for SrcA and unpack a single face for SrcB (scalar), compatible with math reduce column kernels.
 *
 * Supports 32x32 tiles and tiny Nx32 tiles with N <= 8.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @param buf_desc_id_0: Buffer descriptor ID for the operand that will be tilized into srcA
 * @param buf_desc_id_1: Buffer descriptor ID for scaler operand that will be unpacked into srcB
 * @param full_ct_dim: Number of tiles in a row of the input tensor
 * @param tensor_shape: Contains all the information of the tile shape for the input that will be tilized: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE>
inline void _llk_unpack_reduce_col_tilizeA_strided_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t full_ct_dim, const TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);

    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.stride_val_source                  = 0;
    unpk_cfg.f.stride_offset_0                    = full_ct_dim * tensor_shape.num_faces_c_dim; // stride to next row within same tile

    cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
    cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];

    if (tensor_shape.total_num_faces() == NUM_FACES)
    {
        LLK_ASSERT(
            tensor_shape.total_row_dim() == TILE_R_DIM && tensor_shape.total_col_dim() == TILE_C_DIM,
            "Four face unpack reduce col tilizeA strided only supports 32x32 tiles");
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_NO_WRITE_RMW, 0);
        _llk_unpack_reduce_col_tilizeA_strided_mop_config_(buf_desc_id_0, buf_desc_id_1, full_ct_dim, tensor_shape);
    }
    else
    {
        // One strided unpack has to cover a whole face, and both faces of the tile must sit side by side in L1.
        LLK_ASSERT(
            tensor_shape.face_r_dim <= ckernel::unpack::UNPACR_STRIDE_MAX_ROWS && tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM &&
                tensor_shape.total_num_faces() == tensor_shape.num_faces_c_dim,
            "Tiny tile unpack reduce col tilizeA strided only supports Nx32 tiles with N <= 8");

        // A strided unpack writes UNPACR_STRIDE_MAX_ROWS rows; leave the rows past the end of the face untouched so
        // that they keep the pool identity the MOP cleared SrcA to instead of data of the next tile row.
        const std::uint32_t row_mask = (0xFF << tensor_shape.face_r_dim) & 0xFF;
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_NO_WRITE_RMW, 1);
        cfg_rmw(THCON_UNPACKER0_REG1_UNPACK_STRIDE_ROW_MASK1_RMW, row_mask);
        _llk_unpack_reduce_col_tilizeA_strided_tiny_tiles_mop_config_<POOL_TYPE>(buf_desc_id_0, buf_desc_id_1);
    }
}

/**
 * @brief Unpacks and tilizes one tile for SrcA and unpacks a single face for SrcB (scalar), compatible with math reduce column kernels.
 *
 * Drives whichever MOP @ref _llk_unpack_reduce_col_tilizeA_strided_init_ programmed for the tile shape.
 *
 * @param tensor_shape: Contains all the information of the tile shape for the input that will be tilized: num faces, face row/col dim, etc
 * @param start_l1_tile_idx_0: L1 index for UNPACKER0, unpacks to SrcA
 * @param start_l1_tile_idx_1: L1 index for UNPACKER1, unpacks to SrcB
 */
inline void _llk_unpack_reduce_col_tilizeA_strided_(
    const TensorShape& tensor_shape, const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1)
{
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, start_l1_tile_idx_0 * tensor_shape.num_faces_c_dim);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, start_l1_tile_idx_1);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
