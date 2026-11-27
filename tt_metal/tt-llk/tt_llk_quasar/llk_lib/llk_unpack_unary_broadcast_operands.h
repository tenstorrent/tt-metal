// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;

/**
 * @file llk_unpack_unary_broadcast_operands.h
 * @brief UNP replay / MOP setup and per-tile run for unary eltwise with scalar, row, or column broadcast.
 */

/**
 * @brief Builds the MOP for unpacking unary broadcast operands.
 *
 * @tparam UNP_SEL: Unpacker select; must be UNP_B unless unpack_to_dest (then UNP_A), values = <p_unpacr::UNP_A/UNP_B>
 * @tparam BROADCAST_TYPE: Broadcast type, values = <COL/ROW/SCALAR>
 * @tparam EN_32BIT_DEST: True if the Dest register is in 32-bit mode, values = <true/false>
 * @tparam unpack_to_dest: When true, unpack targets math dest (UNP_A); otherwise SrcB (UNP_B), values = <true/false>
 * @tparam acc_to_dest: The math side accumulates the broadcast into dest, so it consumes a zeroed SrcA bank per tile.
 * @param buf_desc_id: Buffer descriptor for the UNPACR source.
 * @param tensor_shape: Face grid and face row dimension. A 32x32 tile is one L1 hardware tile; two-face shapes are one
 *        hardware tile per face and take the per-face path. COL and unpack_to_dest require 32x32.
 * @param num_tiles: Outer MOP loop count (tiles to unpack from L1).
 */
template <std::uint32_t UNP_SEL, BroadcastType BROADCAST_TYPE, bool EN_32BIT_DEST, bool unpack_to_dest = false, bool acc_to_dest = false>
inline void _llk_unpack_unary_broadcast_operands_mop_config_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape, const std::uint32_t num_tiles)
{
    static_assert(!(acc_to_dest && unpack_to_dest), "acc_to_dest is not supported when unpacking to dest");
    static_assert(
        unpack_to_dest || (UNP_SEL == p_unpacr::UNP_B),
        "UNP_SEL must be p_unpacr::UNP_B when unpack_to_dest is false - movA2D broadcast is not working on Quasar");
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool per_face_l1        = num_faces != 1u && num_faces != NUM_FACES;

    const std::uint32_t MOP_OUTER_LOOP            = num_tiles;
    constexpr std::uint32_t MOP_INNER_LOOP        = 1;

    // Fixed lengths for the sequences that stay 32x32 (or are shape independent).
    constexpr std::uint32_t fixed_replay_len = BROADCAST_TYPE == BroadcastType::SCALAR ? 1u : (unpack_to_dest ? 2u : 4u);
    // ROW on SrcB posts one face per tile face: 4 for 32x32, 1 for 16x16.
    const std::uint32_t replay_buf_len = (BROADCAST_TYPE == BroadcastType::ROW && !unpack_to_dest) ? num_faces : fixed_replay_len;

    // Two-face tiles (Nx32 and 32x16) store each face as its own L1 hardware tile (z_dim is 4 only for 32x32),
    // so Src_Face_Idx cannot reach the second face. Read one hardware tile at a time with TT_UNPACR1_TILE_INC,
    // whose increment moves the L1 tile counter after the read.
    if constexpr (!unpack_to_dest && BROADCAST_TYPE != BroadcastType::COL)
    {
        if (per_face_l1)
        {
            const std::uint32_t srcb_posts = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1u : num_faces;
            const bool faces_side_by_side  = tensor_shape.num_faces_c_dim == 2u; // Nx32; otherwise 32x16 (stacked)

            load_replay_buf(
                0u,
                srcb_posts,
                false,
                0,
                0,
                [buf_desc_id, faces_side_by_side]
                {
                    if constexpr (BROADCAST_TYPE == BroadcastType::SCALAR)
                    {
                        // One read of face 0, then skip both faces to reach the next tile.
                        TT_UNPACR1_TILE_INC(0 /*Dst Tile Idx*/, 2 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1);
                    }
                    else if (faces_side_by_side)
                    {
                        // ROW, Nx32: each face is its own column, so read face 0 then face 1.
                        TT_UNPACR1_TILE_INC(0 /*Dst Tile Idx*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1);
                        TT_UNPACR1_TILE_INC(0 /*Dst Tile Idx*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1);
                    }
                    else
                    {
                        // ROW, 32x16: both faces sit in one column, so both read the top face (face 0).
                        TT_UNPACR1_TILE_INC(0 /*Dst Tile Idx*/, 0 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1);
                        TT_UNPACR1_TILE_INC(0 /*Dst Tile Idx*/, 2 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1);
                    }
                });

            // No per-tile increment op: the last TILE_INC already moved the counter to the next tile.
            ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, srcb_posts, 0, 0, 0, 0));
            if constexpr (EN_32BIT_DEST || acc_to_dest)
            {
                // 32-bit dest and acc_to_dest use ELWADD, which requires dvalid from both SrcA and SrcB.
                // Data is unpacked to SrcB, so zero the unused engine (SrcA) and set its dvalid.
                const std::uint32_t clr_unused_unpacr_engine = nop_insn_for_unused_unpacker_engine<UNP_SEL>();
                temp.set_start_op(clr_unused_unpacr_engine);
            }
            temp.program_bank0_sw_cntl(instrn_buffer);
            return;
        }
    }

    if constexpr (BROADCAST_TYPE == BroadcastType::SCALAR)
    {
        load_replay_buf<0, fixed_replay_len>(
            [buf_desc_id]
            {
                if constexpr (unpack_to_dest)
                {
                    TT_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
                }
                else
                {
                    TT_UNPACR1_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                }
            });
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
    {
        if constexpr (unpack_to_dest)
        {
            load_replay_buf<0, fixed_replay_len>(
                [buf_desc_id]
                {
                    TT_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
                    TT_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 1 /*Dst_Face_Idx*/, 1 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
                });
        }
        else
        {
            // Match face order in llk_unpack_binary_broadcast_operands.h (UNP_B / SrcB path): face f uses top face f % num_faces_c_dim.
            load_replay_buf(
                0u,
                replay_buf_len,
                false /*exec_while_loading*/,
                0 /*set_mutex*/,
                0 /*last*/,
                [buf_desc_id, num_faces, tensor_shape]
                {
                    for (std::uint32_t face = 0; face < num_faces; ++face)
                    {
                        TT_UNPACR1_FACE(0 /*Dst Face Idx*/, face % tensor_shape.num_faces_c_dim /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                    }
                });
        }
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
    {
        load_replay_buf<0, fixed_replay_len>(
            [buf_desc_id]
            {
                if constexpr (unpack_to_dest)
                {
                    TT_UNPACR_DEST_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
                    TT_UNPACR_DEST_FACE(2 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
                }
                else
                {
                    // Match face order in llk_unpack_binary_broadcast_operands.h (UNP_B / SrcB path)
                    TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                    TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                    TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                    TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                }
            });
    }

    std::uint32_t inc_src_tile_instrn;
    if constexpr (unpack_to_dest)
    {
        inc_src_tile_instrn = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 1);
    }
    else
    {
        inc_src_tile_instrn = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 1);
    }

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), inc_src_tile_instrn);
    if constexpr (unpack_to_dest)
    {
        temp.set_end_op(TT_OP_INC_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 1));
    }
    else if constexpr (EN_32BIT_DEST || acc_to_dest)
    {
        // 32-bit dest and acc_to_dest use ELWADD, which requires dvalid from both SrcA and SrcB.
        // Data is unpacked to SrcB, so zero the unused engine (SrcA) and set its dvalid.
        const std::uint32_t clr_unused_unpacr_engine = nop_insn_for_unused_unpacker_engine<UNP_SEL>();
        temp.set_start_op(clr_unused_unpacr_engine);
    }
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker for unary operations with broadcasts.
 *
 * @tparam UNP_SEL: Unpacker resource; must be UNP_B unless unpack_to_dest, values = <p_unpacr::UNP_A/UNP_B>
 * @tparam BROADCAST_TYPE: Broadcast type, values = <COL/ROW/SCALAR>
 * @tparam EN_32BIT_DEST: True if the Dest register is in 32-bit mode, values = <true/false>
 * @tparam unpack_to_dest: Route unpack to dest (UNP_A) vs SrcB (UNP_B), values = <true/false>
 * @tparam acc_to_dest: The math side accumulates the broadcast into dest; not supported with unpack_to_dest.
 * @param buf_desc_id: Buffer descriptor for the UNPACR source.
 * @param tensor_shape: Shape of the broadcast operand tile. Pass the same shape to the matching execute call.
 * @param num_tiles: Number of tiles in the outer unpack loop.
 * @note On the math thread, pair with @ref _llk_math_eltwise_unary_broadcast_init_ (T1) with matching
 *       BROADCAST_TYPE/EN_32BIT_DEST/unpack_to_dest/acc_to_dest.
 * @note With acc_to_dest, configure the SrcA unpacker format as well as SrcB (for example with _llk_unpack_configure_binary_).
 * @note @ref _llk_unpack_unary_broadcast_operands_ is the matching execute call on this thread.
 * @note COL broadcast and unpack_to_dest support only 32x32 tiles; other shapes fail an LLK_ASSERT.
 */
template <std::uint32_t UNP_SEL, BroadcastType BROADCAST_TYPE, bool EN_32BIT_DEST, bool unpack_to_dest = false, bool acc_to_dest = false>
inline void _llk_unpack_unary_broadcast_operands_init_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape, const std::uint32_t num_tiles)
{
    const bool full_tile =
        tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM && tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM;
    LLK_ASSERT(BROADCAST_TYPE != BroadcastType::COL || full_tile, "Unary column broadcast requires 32x32 tiles");
    LLK_ASSERT(!unpack_to_dest || full_tile, "Unary broadcast with unpack_to_dest requires 32x32 tiles");
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_unary_broadcast_operands_init_", tensor_shape);

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_unary_broadcast_operands_mop_config_<UNP_SEL, BROADCAST_TYPE, EN_32BIT_DEST, unpack_to_dest, acc_to_dest>(buf_desc_id, tensor_shape, num_tiles);
}

/**
 * @brief Runs the bank0 unpack MOP for one invocation (sets tile/face indices then executes the replay).
 *
 * @tparam UNP_SEL: Logical unpack select for static checks; counter uses UNP_A when unpack_to_dest, values = <p_unpacr::UNP_A/UNP_B>
 * @tparam unpack_to_dest: Use the UNP_A dest path vs the UNP_B SrcB path, values = <true/false>
 * @param start_l1_tile_idx: Starting tile index in L1, counted in tiles of tensor_shape.
 * @param tensor_shape: Must match the shape passed to init. 32x32 is one hardware tile per tile; two-face shapes are
 *        one hardware tile per face, so the start index is scaled by the face count.
 * @note Call @ref _llk_unpack_unary_broadcast_operands_init_ with matching template args before this function.
 */
template <std::uint32_t UNP_SEL, bool unpack_to_dest = false>
inline void _llk_unpack_unary_broadcast_operands_(const std::uint32_t start_l1_tile_idx, const TensorShape& tensor_shape)
{
    static_assert(
        unpack_to_dest || (UNP_SEL == p_unpacr::UNP_B),
        "UNP_SEL must be p_unpacr::UNP_B when unpack_to_dest is false - movA2D broadcast is not working on Quasar");

    constexpr std::uint32_t counter_unp_sel = unpack_to_dest ? p_unpacr::UNP_A : UNP_SEL;
    const std::uint32_t hw_tiles_per_sw_tile = tensor_shape.total_num_faces() == NUM_FACES ? 1u : tensor_shape.total_num_faces();
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, counter_unp_sel, start_l1_tile_idx * hw_tiles_per_sw_tile);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, counter_unp_sel, 0);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
