// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"
using namespace ckernel;

/**
 * @brief Builds the MOP for unpacking binary operands (SrcA and SrcB) tile by tile.
 *
 * buf_desc_id_0 feeds UNPACKER0 -> SRCA, buf_desc_id_1 feeds UNPACKER1 -> SRCB.
 *
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: Number of tiles to unpack at a time for both inputs.
 * @param tensor_shape: Shape shared by both operands.
 */
inline void _llk_unpack_binary_operands_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t num_tiles, const TensorShape& tensor_shape)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    if (num_faces != NUM_FACES && num_faces != 1)
    {
        // construct_buf_desc maps two-face SW tiles to one-face HW tiles (z_dim=1);
        // the descriptor cannot encode z_dim=2. Unpack each face separately.
        // L1 faces are contiguous; register-file faces use the padded FPU-row stride.
        const std::uint32_t dest_tile_idx_inc = tiny_face_stride(tensor_shape);

        const std::uint32_t MOP_INNER_LOOP = num_faces - 1;
        ckernel_template temp(
            num_tiles,
            MOP_INNER_LOOP,
            TT_OP_UNPACR0_TILE_INC(dest_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id_0, 0 /*Set Dvalid*/),
            TT_OP_UNPACR1_TILE_INC(dest_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id_1, 0 /*Set Dvalid*/));
        if (num_tiles > 1)
        {
            // Reset both unpacker destination counters for each SW tile in a batch.
            // Single-tile calls use the counter reset in _llk_unpack_binary_operands_.
            load_replay_buf<0, 2>(
                []
                {
                    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
                    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);
                });
            temp.set_start_op(TT_OP_REPLAY(0, 2, 0, 0, 0, 0));
        }
        // Publish each operand only after its last face is loaded.
        temp.set_end_ops(
            TT_OP_UNPACR0_TILE_INC(dest_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/),
            TT_OP_UNPACR1_TILE_INC(dest_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id_1, 1 /*Set Dvalid*/));
        temp.program_bank0_sw_cntl(instrn_buffer);
        return;
    }

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = num_tiles;

    std::uint32_t unpack_instrn0 = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    std::uint32_t unpack_instrn1 = TT_OP_UNPACR1_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_1, 1 /*Set Dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_instrn0, unpack_instrn1);

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
 * @param tensor_shape: Shape shared by both operands.
 * @note On the math thread, pair with @ref _llk_math_eltwise_binary_init_ (T1); on the pack thread, pair with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_unpack_binary_operands_ is the matching execute call on this thread.
 * @note Buffer descriptors must match construct_buf_desc(tensor_shape): x_dim=16, y_dim=face_r_dim,
 *       z_dim=1 for one/two-face software tiles or z_dim=4 for full 32x32 tiles.
 *       Four-face tiles require 16-row faces, as required by validate_buffer_desc.
 */
inline void _llk_unpack_binary_operands_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape& tensor_shape, const std::uint32_t num_tiles = NUM_TILES)
{
    LLK_ASSERT(tensor_shape.total_num_faces() != NUM_FACES || tensor_shape.face_r_dim == MAX_FACE_R_DIM, "Binary unpack four-face tiles require 16-row faces");
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_binary_operands_init_", tensor_shape);
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_binary_operands_mop_config_(buf_desc_id_0, buf_desc_id_1, num_tiles, tensor_shape);
}

/**
 * @brief Unpacks binary operands into SrcA and SrcB.
 *
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer;
 *        start_l1_tile_idx_0 -> UNPACKER0 -> SRCA, start_l1_tile_idx_1 -> UNPACKER1 -> SRCB.
 * @param tensor_shape: Shape shared by both operands; must match init.
 * @note Call @ref _llk_unpack_binary_operands_init_ with matching template args before this function.
 */
inline void _llk_unpack_binary_operands_(const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1, const TensorShape& tensor_shape)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    // Reset Dest counters for Unpacker0/1 to 0
    // Set Source counter to L1 base + offset
    const std::uint32_t hw_tiles_per_sw_tile = tensor_shape.total_num_faces() == NUM_FACES ? 1 : tensor_shape.total_num_faces();
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, start_l1_tile_idx_0 * hw_tiles_per_sw_tile);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, start_l1_tile_idx_1 * hw_tiles_per_sw_tile);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
