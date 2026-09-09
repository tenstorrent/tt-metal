// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"
#include "tensor_shape.h"
using namespace ckernel;

/**
 * @brief Builds the MOP for unpacking binary operands (SrcA and SrcB) tile by tile.
 *
 * buf_desc_id_0 feeds UNPACKER0 -> SRCA, buf_desc_id_1 feeds UNPACKER1 -> SRCB.
 * Used for 32x32 (z_dim=4, one UNPACR per SW tile) and 16x16 (one face per SW tile).
 *
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: Number of tiles to unpack at a time for both inputs.
 */
inline void _llk_unpack_binary_operands_mop_config_(const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t num_tiles)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = num_tiles;

    std::uint32_t unpack_instrn0 = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    std::uint32_t unpack_instrn1 = TT_OP_UNPACR1_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_1, 1 /*Set Dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn0, unpack_instrn1);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Builds the MOP for unpacking multi-face tiny tiles into SrcA and SrcB.
 *
 * Each SW tile is multiple HW tiles (one face each). Faces are unpacked into consecutive
 * Src slots (sparse 8-row dest occupancy when face_r_dim < 8). Dvalid is raised only on
 * the last face so math consumes one SrcA/SrcB valid per SW tile.
 */
inline void _llk_unpack_binary_operands_variable_tile_size_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t num_tiles, const TensorShape& tensor_shape)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    LLK_ASSERT(num_faces > 1, "variable-tile binary unpack is for multi-face tiny tiles");

    const std::uint32_t dest_tile_idx_inc = (static_cast<std::uint32_t>(tensor_shape.face_r_dim) < (FACE_R_DIM >> 1))
                                                ? (FACE_R_DIM >> (rows_log2(static_cast<std::uint32_t>(tensor_shape.face_r_dim)) + 1))
                                                : 1;

    const std::uint32_t unpack_instrn0    = TT_OP_UNPACR0_TILE_INC(dest_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id_0, 0 /*Set Dvalid*/);
    const std::uint32_t unpack_instrn1    = TT_OP_UNPACR1_TILE_INC(dest_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id_1, 0 /*Set Dvalid*/);
    const std::uint32_t unpack_instrn0_dv = TT_OP_UNPACR0_TILE_INC(dest_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    const std::uint32_t unpack_instrn1_dv = TT_OP_UNPACR1_TILE_INC(dest_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id_1, 1 /*Set Dvalid*/);

    // Inner loop covers faces 0..N-2; last face + dvalid runs as END_OPs (same pattern as unary tiny-tile unpack).
    ckernel_template temp(num_tiles, num_faces - 1, unpack_instrn0, unpack_instrn1);
    temp.set_end_ops(unpack_instrn0_dv, unpack_instrn1_dv);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker for binary operations (SrcA and SrcB).
 *
 * Programs the MOP for unpacking binary operands. buf_desc_id_0 feeds UNPACKER0 -> SRCA,
 * buf_desc_id_1 feeds UNPACKER1 -> SRCB.
 *
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: Number of tiles to unpack at a time for both inputs.
 * @param tensor_shape: Tile shape; multi-face tiny tiles (2/4/8/16x32, 32x16) use the face-loop MOP.
 * @note On the math thread, pair with @ref _llk_math_eltwise_binary_init_ (T1); on the pack thread, pair with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_unpack_binary_operands_ is the matching execute call on this thread.
 */
inline void _llk_unpack_binary_operands_init_(
    const std::uint32_t buf_desc_id_0,
    const std::uint32_t buf_desc_id_1,
    const std::uint32_t num_tiles   = NUM_TILES,
    const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    if (tensor_shape.total_num_faces() == NUM_FACES || tensor_shape.total_num_faces() == 1)
    {
        _llk_unpack_binary_operands_mop_config_(buf_desc_id_0, buf_desc_id_1, num_tiles);
    }
    else
    {
        _llk_unpack_binary_operands_variable_tile_size_mop_config_(buf_desc_id_0, buf_desc_id_1, num_tiles, tensor_shape);
    }
}

/**
 * @brief Unpacks binary operands into SrcA and SrcB.
 *
 * @param start_l1_tile_idx_0/1: Start SW tile index into the L1 buffer;
 *        start_l1_tile_idx_0 -> UNPACKER0 -> SRCA, start_l1_tile_idx_1 -> UNPACKER1 -> SRCB.
 * @param tensor_shape: Must match the shape passed to @ref _llk_unpack_binary_operands_init_.
 * @note Call @ref _llk_unpack_binary_operands_init_ with matching args before this function.
 */
inline void _llk_unpack_binary_operands_(
    const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1, const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    // Tiny tiles are one HW tile per face; scale the SW tile index into a face index.
    const std::uint32_t l1_tile_idx_0 =
        (tensor_shape.total_num_faces() == NUM_FACES) ? start_l1_tile_idx_0 : start_l1_tile_idx_0 * tensor_shape.total_num_faces();
    const std::uint32_t l1_tile_idx_1 =
        (tensor_shape.total_num_faces() == NUM_FACES) ? start_l1_tile_idx_1 : start_l1_tile_idx_1 * tensor_shape.total_num_faces();

    // Reset Dest counters for Unpacker0/1 to 0
    // Set Source counter to L1 base + offset
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx_0);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, l1_tile_idx_1);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
