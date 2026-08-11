// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "tensor_shape.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @file llk_math_unary_broadcast.h
 * @brief Math addrmods, MOP, and per-tile run for unary eltwise with scalar, row, or column broadcast.
 */

/**
 * @brief Programs address modifiers for eltwise unary broadcast (MOVB2D / MOVD2B paths).
 *
 * @tparam BROADCAST_TYPE: Scalar, row, or column broadcast (must not be NONE), values = <COL/ROW/SCALAR>
 * @tparam unpack_to_dest: When true, UNP_A wrote to dest; ADDR_MOD_3 used for dest<->srcB moves
 * @param tensor_shape: Face geometry (face_r_dim, num_faces) for row-broadcast addrmods
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest>
inline void _llk_math_eltwise_unary_broadcast_addrmod_(const TensorShape& tensor_shape)
{
    static_assert(BROADCAST_TYPE != BroadcastType::NONE, "Broadcast type cannot be NONE");

    constexpr std::uint16_t row_step    = static_cast<std::uint16_t>(ELTWISE_MATH_ROWS);
    constexpr std::uint8_t srcb_col_inc = (BROADCAST_TYPE == BroadcastType::COL) ? static_cast<std::uint8_t>(ELTWISE_MATH_ROWS) : static_cast<std::uint8_t>(0);

    addr_mod_t {.srcb = {.incr = srcb_col_inc}, .dest = {.incr = row_step}}.set(ADDR_MOD_0);
    addr_mod_t {.srcb = {.clr = 1}, .dest = {.incr = row_step}}.set(ADDR_MOD_1);

    if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
    {
        addr_mod_t {
            .srcb = {.incr = static_cast<std::uint8_t>(tensor_shape.face_r_dim)},
            .dest = {.incr = row_step},
        }
            .set(ADDR_MOD_2);
    }

    if constexpr (unpack_to_dest)
    {
        if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
        {
            addr_mod_t {
                .srcb = {.incr = 0},
                .dest = {.incr = static_cast<std::uint16_t>(tensor_shape.face_r_dim)},
            }
                .set(ADDR_MOD_3);
        }
        else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
        {
            addr_mod_t {
                .srcb = {.incr = 0},
                .dest = {.incr = static_cast<std::uint16_t>(tensor_shape.face_r_dim * tensor_shape.num_faces_c_dim)},
            }
                .set(ADDR_MOD_3);
        }
        else
        {
            addr_mod_t {
                .srcb = {.incr = 0},
                .dest = {.incr = row_step},
            }
                .set(ADDR_MOD_3);
        }

        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
        }
            .set(ADDR_MOD_4);
    }
}

/**
 * @brief MOP / replay configuration for unary math broadcast.
 *
 * @tparam BROADCAST_TYPE: Scalar, row, or column broadcast, values = <COL/ROW/SCALAR>
 * @tparam unpack_to_dest: When true, unpack filled dest; MOVB2D reads srcB only, so @ref _llk_math_eltwise_unary_broadcast_ runs MOVD2B (dest->srcB) first,
 *         then programs this MOVB2D MOP. ROW uses the fixed replay below; COL/SCALAR use the same outer/inner template as the non-unpack_to_dest path with
 *         ADDR_MOD_3 from addrmod.
 * @param tensor_shape: Tile shape for loop counts and row dimensions
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest>
inline void _llk_math_eltwise_unary_broadcast_mop_config_(const TensorShape& tensor_shape)
{
    static_assert(BROADCAST_TYPE != BroadcastType::NONE, "Broadcast type cannot be NONE");

    if constexpr (unpack_to_dest)
    {
        if constexpr (BROADCAST_TYPE == BroadcastType::COL)
        {
            constexpr std::uint32_t replay_buf_len = 12;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // Read F0/F2 hi16 from DEST → SrcB[0:15]
                    TTI_MOVD2B(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_OFF, 0);
                    TTI_MOVD2B(p_mov::DEST_NORM, 8, ADDR_MOD_4, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_OFF, 8);

                    // Read F0/F2 lo16 from DEST → SrcB[16:31]
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 16, ADDR_MOD_4, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_OFF, 0);
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 24, ADDR_MOD_4, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_OFF, 8);

                    // Write hi16 to DEST F0,F1/F2,F3 from SrcB[0:31] (column broadcast ON)
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 0);
                    TTI_MOVB2D(p_mov::DEST_NORM, 8, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 8);
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 16);
                    TTI_MOVB2D(p_mov::DEST_NORM, 8, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 24);

                    // Write lo16 to DEST F0,F1/F2,F3 from SrcB[0:31] (column broadcast ON)
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 16, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 0);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 24, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 8);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 16, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 16);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 24, ADDR_MOD_3, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 24); // dst += 2* face_r_dim, F0 → F2
                });

            ckernel_template temp(1 /* mop_outer_loop */, 2 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
        {
            constexpr std::uint32_t replay_buf_len = 10;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // Read F0/F1 rows[0:7] hi16 and lo16 from DEST → SrcB[0:15]
                    TTI_MOVD2B(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_OFF, 0);
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 8, ADDR_MOD_4, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_OFF, 0);

                    // Write hi16 to DEST F0,F2/F1,F3 from SrcB[0:7] (row broadcast ON)
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0 + 1);
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8 + 1);
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 32 + 1);
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 40 + 1);

                    // Write lo16 to DEST F0,F2/F1,F3 from SrcB[8:15] (row broadcast ON)
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 0 + 1);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 8 + 1);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 32 + 1);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_3, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_OFF, 40 + 1); // dst += face_r_dim, F0 → F1
                });

            ckernel_template temp(1 /* mop_outer_loop */, 2 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else // BroadcastType::SCALAR
        {
            constexpr std::uint32_t replay_buf_len = 4;
            load_replay_buf<0, replay_buf_len>(
                []
                {
                    // Read F0 rows[0:7] hi16 and lo16 from DEST → SrcB[0:15]
                    TTI_MOVD2B(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_OFF, 0);
                    TTI_MOVD2B(p_mov::DEST_32B_LOW, 8, ADDR_MOD_4, p_movd2b::MOV_8_ROWS, p_movd2b::TRANSPOSE_OFF, 0);

                    // Write hi16 and lo16 to DEST[0:63] from SrcB[0:15] (row and column broadcast ON)
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 0 + 1);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, 8, ADDR_MOD_3, p_mov_src_to_dest::MOV_8_ROWS, p_movb2d::BCAST_ON, 0 + 1); // dst += 8
                });

            ckernel_template temp(1 /* mop_outer_loop */, 8 /* mop_inner_loop */, TT_OP_REPLAY(2, 2, 0, 0, 0, 0));
            temp.set_start_op(TT_OP_REPLAY(0, 2, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
    }
    else
    {
        const std::uint32_t num_rows =
            (BROADCAST_TYPE == BroadcastType::SCALAR) ? tensor_shape.total_num_faces() * tensor_shape.face_r_dim : tensor_shape.face_r_dim;
        const std::uint32_t outer = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1U : static_cast<std::uint32_t>(tensor_shape.total_num_faces());
        const std::uint32_t inner = num_rows >> rows_log2(ELTWISE_MATH_ROWS);

        const std::uint32_t bcast_row     = (BROADCAST_TYPE != BroadcastType::COL) ? 1U : 0U;
        constexpr std::uint32_t bcast_col = (BROADCAST_TYPE != BroadcastType::ROW) ? 1U : 0U;

        const auto bcast_instr = [bcast_col, bcast_row](std::uint8_t addr_mod)
        {
            return TT_OP_MOVB2D(0, 0, addr_mod, p_mov_src_to_dest::MOV_8_ROWS, bcast_col, bcast_row /* dst_addr */);
        }; // adding 1 to dst_addr enables row broadcast

        ckernel_template temp(outer, inner, bcast_instr(ADDR_MOD_0));
        temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
        if constexpr (BROADCAST_TYPE == BroadcastType::SCALAR)
        {
            temp.set_last_outer_loop_instr(bcast_instr(ADDR_MOD_1));
        }
        else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
        {
            temp.set_last_inner_loop_instr(bcast_instr(ADDR_MOD_1));
        }
        else
        {
            temp.set_last_inner_loop_instr(bcast_instr(ADDR_MOD_0));
            temp.set_last_outer_loop_instr(bcast_instr(ADDR_MOD_2));
        }

        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Init unary-broadcast math: addrmods, mop configuration, reset counters.
 *
 * @tparam BROADCAST_TYPE: Scalar, row, or column broadcast, values = <COL/ROW/SCALAR>
 * @tparam unpack_to_dest: UNP path wrote to dest
 * @param tensor_shape: Passed to addrmod / MOP setup
 * @note On the unpack thread, pair with @ref _llk_unpack_unary_broadcast_operands_init_ (T0) with matching BROADCAST_TYPE/unpack_to_dest.
 * @note @ref _llk_math_eltwise_unary_broadcast_ runs the configured op with matching template args.
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest>
inline void _llk_math_eltwise_unary_broadcast_init_(const TensorShape& tensor_shape)
{
    LLK_ASSERT(
        tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM && tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM,
        "Unary broadcast currently only supports 32x32 tiles (face_r_dim=16, 2x2 faces)");

    _llk_math_eltwise_unary_broadcast_addrmod_<BROADCAST_TYPE, unpack_to_dest>(tensor_shape);
    _llk_math_eltwise_unary_broadcast_mop_config_<BROADCAST_TYPE, unpack_to_dest>(tensor_shape);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Run one tile of unary broadcast math: set dest write addr
 *
 * @param tile_idx: Destination tile index within current dest bank (SyncHalf)
 * @note Call @ref _llk_math_eltwise_unary_broadcast_init_ with matching template args before this function.
 */
inline void _llk_math_eltwise_unary_broadcast_(const std::uint32_t tile_idx)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);

    // Wait condition SRCB_VLD is required as MOVD2B doesn't automatically wait
    // for SrcB[MatrixUnit.SrcBBank].AllowedClient == SrcClient::MatrixUnit.
    TTI_STALLWAIT(p_stall::STALL_MATH, 0, p_stall::WAIT_SFPU, p_stall::SRCB_VLD); // TEN-4367 - SrcB sync workaround

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}
