// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"
#include "tensor_shape.h"
using namespace ckernel;

/**
 * @brief Builds the MOP for unpacking binary operands with broadcast (SrcA and SrcB), tile by tile.
 *
 * Broadcast only operates on the SrcB register. buf_desc_id_0 feeds UNPACKER0 -> SRCA,
 * buf_desc_id_1 feeds UNPACKER1 -> SRCB.
 *
 * @tparam BROADCAST_TYPE: Broadcast type for SrcB, values = <COL/ROW/SCALAR>
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: Number of tiles to unpack at a time for both inputs.
 */
template <BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_binary_broadcast_operands_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape& tensor_shape, const std::uint32_t num_tiles)
{
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");

    const std::uint32_t num_faces          = tensor_shape.total_num_faces();
    const std::uint32_t MOP_OUTER_LOOP     = num_tiles;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;
    // z_dim is 4 only for a full 32x32 tile. Every other shape is one hardware tile per face.
    const bool per_face_l1 = num_faces != 1u && num_faces != NUM_FACES;

    auto srcb_face_index = [tensor_shape](std::uint32_t face) -> std::uint32_t
    {
        if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
        {
            return face % tensor_shape.num_faces_c_dim;
        }
        else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
        {
            return (face / tensor_shape.num_faces_c_dim) * tensor_shape.num_faces_c_dim;
        }
        return 0u;
    };

    if (per_face_l1)
    {
        // Each face is its own L1 hardware tile. Replay SrcA faces and raise one dvalid on the last,
        // and replay SrcB from the broadcast source face of that same tile.
        const std::uint32_t dest_stride = tiny_face_stride(tensor_shape);
        const std::uint32_t srcb_posts  = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1u : num_faces;
        // One unpack per posted face. The source increment on that unpack steps L1 to the next face.
        const std::uint32_t srcb_replay_len = srcb_posts;

        const std::uint32_t srca_replay_len   = 1u + num_faces;
        const std::uint32_t srca_replay_start = srcb_replay_len;

        load_replay_buf(
            0u,
            srcb_replay_len,
            false,
            0,
            0,
            [=]
            {
                for (std::uint32_t face = 0; face < srcb_posts; ++face)
                {
                    const std::uint32_t src_face = srcb_face_index(face);
                    const std::uint32_t next_src = (face + 1u < srcb_posts) ? srcb_face_index(face + 1u) : num_faces;
                    // Math re-reads SrcB at row 0. The source increment runs after the read and
                    // lands on the next face, or on the next software tile after the last post.
                    TT_UNPACR1_TILE_INC(0 /*Dst Tile Idx*/, next_src - src_face, buf_desc_id_1, 1 /*Set Dvalid*/);
                }
            });

        load_replay_buf(
            srca_replay_start,
            srca_replay_len,
            false,
            0,
            0,
            [=]
            {
                TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
                for (std::uint32_t face = 0; face < num_faces; ++face)
                {
                    const bool last_face = face + 1u == num_faces;
                    TT_UNPACR0_TILE_INC(dest_stride, 1 /*Src Tile Idx*/, buf_desc_id_0, last_face ? 1u : 0u /*Set Dvalid*/);
                }
            });

        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, srcb_replay_len, 0, 0, 0, 0));
        temp.set_start_op(TT_OP_REPLAY(srca_replay_start, srca_replay_len, 0, 0, 0, 0));
        temp.program_bank0_sw_cntl(instrn_buffer);
        return;
    }

    const std::uint32_t unpack_srca_tile_inc = TT_OP_UNPACR0_TILE_INC(0, 1 /*Src Tile Idx*/, buf_desc_id_0, 1 /*Set Dvalid*/);
    // One SrcB dvalid per face inside a hardware tile. 16x16 has one face; 32x32 has four.
    const std::uint32_t replay_buf_len = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1u : num_faces;

    load_replay_buf(
        0u,
        replay_buf_len,
        false,
        0,
        0,
        [buf_desc_id_1, replay_buf_len, srcb_face_index]
        {
            for (std::uint32_t face = 0; face < replay_buf_len; ++face)
            {
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, srcb_face_index(face), 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
            }
        });

    ckernel_template temp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0),
        TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 1)); // Inc SrcB by 1 tile, because above UNPACR1_FACE does not inc counters

    temp.set_start_op(unpack_srca_tile_inc);

    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker for binary broadcast operations (SrcA and SrcB).
 *
 * Programs the MOP for unpacking binary operands with broadcast. Broadcast only operates on the SrcB
 * register. buf_desc_id_0 feeds UNPACKER0 -> SRCA, buf_desc_id_1 feeds UNPACKER1 -> SRCB.
 *
 * @tparam BROADCAST_TYPE: Broadcast type for SrcB, values = <COL/ROW/SCALAR>
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param num_tiles: Number of tiles to unpack at a time for both inputs.
 * @note On the math thread, pair with @ref _llk_math_eltwise_binary_broadcast_init_ (T1) with matching BROADCAST_TYPE; on the pack thread, pair with
 *       @ref _llk_pack_init_ (T2).
 * @note @ref _llk_unpack_binary_broadcast_operands_ is the matching execute call on this thread.
 */
template <BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_binary_broadcast_operands_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape& tensor_shape, const std::uint32_t num_tiles = NUM_TILES)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_binary_broadcast_operands_mop_config_<BROADCAST_TYPE>(buf_desc_id_0, buf_desc_id_1, tensor_shape, num_tiles);
}

// Full-tile entry point used by fused kernels that do not pass a tile shape.
template <BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_binary_broadcast_operands_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const std::uint32_t num_tiles = NUM_TILES)
{
    _llk_unpack_binary_broadcast_operands_init_<BROADCAST_TYPE>(buf_desc_id_0, buf_desc_id_1, ckernel::DEFAULT_TENSOR_SHAPE, num_tiles);
}

/**
 * @brief Unpacks binary broadcast operands into SrcA and SrcB.
 *
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer;
 *        start_l1_tile_idx_0 -> UNPACKER0 -> SRCA, start_l1_tile_idx_1 -> UNPACKER1 -> SRCB.
 * @param tensor_shape: Shape shared by both operands. 32x32 keeps one hardware tile per software tile; other shapes count L1 in faces.
 * @note Call @ref _llk_unpack_binary_broadcast_operands_init_ with matching template args before this function.
 */
inline void _llk_unpack_binary_broadcast_operands_(
    const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1, const TensorShape& tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    // Reset Dest counters for Unpacker0/1 to 0
    // Set Source counter to L1 base + offset
    const std::uint32_t hw_tiles_per_sw_tile = tensor_shape.total_num_faces() == NUM_FACES ? 1u : tensor_shape.total_num_faces();
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, start_l1_tile_idx_0 * hw_tiles_per_sw_tile);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, start_l1_tile_idx_1 * hw_tiles_per_sw_tile);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
