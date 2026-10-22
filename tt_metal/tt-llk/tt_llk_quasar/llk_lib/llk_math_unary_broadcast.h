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
 * @tparam EN_32BIT_DEST: True if the Dest register is in 32-bit mode, values = <true/false>
 * @tparam unpack_to_dest: When true, record MOVD2B reads from dest followed by MOVB2D broadcast writes in the replay buffer.
 * @param tensor_shape: Tile shape for loop counts and row dimensions
 */
template <BroadcastType BROADCAST_TYPE, bool EN_32BIT_DEST, bool unpack_to_dest>
inline void _llk_math_eltwise_unary_broadcast_mop_config_(const TensorShape& tensor_shape)
{
    static_assert(BROADCAST_TYPE != BroadcastType::NONE, "Broadcast type cannot be NONE");

    if constexpr (unpack_to_dest)
    {
        constexpr std::uint32_t MOVS_PER_FACE = FACE_R_DIM / ELTWISE_MATH_ROWS;
        // Row/scalar broadcast caches eight source rows, independently of the FPU width.
        constexpr std::uint32_t SOURCE_ROWS = 8;
        constexpr std::uint32_t READ_MOVS   = 2 * SOURCE_ROWS / ELTWISE_MATH_ROWS;

        if constexpr (BROADCAST_TYPE == BroadcastType::COL)
        {
            constexpr std::uint32_t replay_buf_len = 6 * MOVS_PER_FACE;
            load_replay_buf<0, replay_buf_len>(
                []
                {
            // Cache F0/F2 hi16 in SrcB[0:15] and lo16 in SrcB[16:31].
#pragma GCC unroll 4
                    for (const auto row : fpu_row_offsets<FACE_R_DIM>())
                    {
                        TTI_MOVD2B(p_mov::DEST_NORM, row, ADDR_MOD_4, FPU_MOV_ROWS, p_movd2b::TRANSPOSE_OFF, row);
                        TTI_MOVD2B(p_mov::DEST_32B_LOW, FACE_R_DIM + row, ADDR_MOD_4, FPU_MOV_ROWS, p_movd2b::TRANSPOSE_OFF, row);
                    }

                // Broadcast both planes to F0,F1/F2,F3 after all source rows are cached.
#pragma GCC unroll 4
                    for (const auto row : fpu_row_offsets<FACE_R_DIM>())
                    {
                        TTI_MOVB2D(p_mov::DEST_NORM, row, ADDR_MOD_4, FPU_MOV_ROWS, p_movb2d::BCAST_ON, row);
                        TTI_MOVB2D(p_mov::DEST_NORM, row, ADDR_MOD_4, FPU_MOV_ROWS, p_movb2d::BCAST_ON, FACE_R_DIM + row);
                        TTI_MOVB2D(p_mov::DEST_32B_LOW, FACE_R_DIM + row, ADDR_MOD_4, FPU_MOV_ROWS, p_movb2d::BCAST_ON, row);

                        // Advance from F0 to F2 after the final MOV.
                        const std::uint8_t addr_mod = row + ELTWISE_MATH_ROWS == FACE_R_DIM ? ADDR_MOD_3 : ADDR_MOD_4;
                        TTI_MOVB2D(p_mov::DEST_32B_LOW, FACE_R_DIM + row, addr_mod, FPU_MOV_ROWS, p_movb2d::BCAST_ON, FACE_R_DIM + row);
                    }
                });

            ckernel_template temp(1 /* mop_outer_loop */, 2 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
        {
            constexpr std::uint32_t replay_buf_len = READ_MOVS + 4 * MOVS_PER_FACE;
            load_replay_buf<0, replay_buf_len>(
                []
                {
            // Read F0/F1 rows[0:7] hi16 and lo16 from DEST → SrcB[0:15].
#pragma GCC unroll 4
                    for (const auto row : fpu_row_offsets<SOURCE_ROWS>())
                    {
                        TTI_MOVD2B(p_mov::DEST_NORM, row, ADDR_MOD_4, FPU_MOV_ROWS, p_movd2b::TRANSPOSE_OFF, row);
                        TTI_MOVD2B(p_mov::DEST_32B_LOW, SOURCE_ROWS + row, ADDR_MOD_4, FPU_MOV_ROWS, p_movd2b::TRANSPOSE_OFF, row);
                    }

                // Broadcast the cached hi16/lo16 source rows to F0,F2/F1,F3.
#pragma GCC unroll 4
                    for (const auto row : fpu_row_offsets<FACE_R_DIM>())
                    {
                        TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, FPU_MOV_ROWS, p_movb2d::BCAST_OFF, row + 1);
                        TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, FPU_MOV_ROWS, p_movb2d::BCAST_OFF, 2 * FACE_R_DIM + row + 1);
                        TTI_MOVB2D(p_mov::DEST_32B_LOW, SOURCE_ROWS, ADDR_MOD_4, FPU_MOV_ROWS, p_movb2d::BCAST_OFF, row + 1);

                        // Advance from F0 to F1 after the final MOV.
                        const std::uint8_t addr_mod = row + ELTWISE_MATH_ROWS == FACE_R_DIM ? ADDR_MOD_3 : ADDR_MOD_4;
                        TTI_MOVB2D(p_mov::DEST_32B_LOW, SOURCE_ROWS, addr_mod, FPU_MOV_ROWS, p_movb2d::BCAST_OFF, 2 * FACE_R_DIM + row + 1);
                    }
                });

            ckernel_template temp(1 /* mop_outer_loop */, 2 /* mop_inner_loop */, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else // BroadcastType::SCALAR
        {
            constexpr std::uint32_t replay_buf_len = READ_MOVS + 2;
            load_replay_buf<0, replay_buf_len>(
                []
                {
            // Read F0 rows[0:7] hi16 and lo16 from DEST → SrcB[0:15].
#pragma GCC unroll 4
                    for (const auto row : fpu_row_offsets<SOURCE_ROWS>())
                    {
                        TTI_MOVD2B(p_mov::DEST_NORM, row, ADDR_MOD_4, FPU_MOV_ROWS, p_movd2b::TRANSPOSE_OFF, row);
                        TTI_MOVD2B(p_mov::DEST_32B_LOW, SOURCE_ROWS + row, ADDR_MOD_4, FPU_MOV_ROWS, p_movd2b::TRANSPOSE_OFF, row);
                    }

                    // The MOP replays only these two writes after caching the source once.
                    TTI_MOVB2D(p_mov::DEST_NORM, 0, ADDR_MOD_4, FPU_MOV_ROWS, p_movb2d::BCAST_ON, 0 + 1);
                    TTI_MOVB2D(p_mov::DEST_32B_LOW, SOURCE_ROWS, ADDR_MOD_3, FPU_MOV_ROWS, p_movb2d::BCAST_ON, 0 + 1);
                });

            ckernel_template temp(1 /* mop_outer_loop */, NUM_FACES * MOVS_PER_FACE /* mop_inner_loop */, TT_OP_REPLAY(READ_MOVS, 2, 0, 0, 0, 0));
            temp.set_start_op(TT_OP_REPLAY(0, READ_MOVS, 0, 0, 0, 0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
    }
    else
    {
        // ELWADD (32-bit dest) or MOVB2D
        const std::uint32_t num_faces     = static_cast<std::uint32_t>(tensor_shape.total_num_faces());
        const std::uint32_t rows_per_face = tensor_shape.face_r_dim >> rows_log2(ELTWISE_MATH_ROWS);
        const std::uint32_t outer         = (EN_32BIT_DEST || BROADCAST_TYPE != BroadcastType::SCALAR) ? num_faces : 1U;
        const std::uint32_t inner         = (EN_32BIT_DEST || BROADCAST_TYPE != BroadcastType::SCALAR) ? rows_per_face : num_faces * rows_per_face;

        if constexpr (EN_32BIT_DEST)
        {
            constexpr auto srcb_bcast                   = (BROADCAST_TYPE == BroadcastType::COL)   ? p_elwise::SRCB_BCAST_COL
                                                          : (BROADCAST_TYPE == BroadcastType::ROW) ? p_elwise::SRCB_BCAST_ROW
                                                                                                   : p_elwise::SRCB_BCAST_ALL;
            constexpr std::uint32_t last_inner_addr_mod = (BROADCAST_TYPE == BroadcastType::COL) ? ADDR_MOD_1 : ADDR_MOD_0;
            const auto elwadd_bcast_instr               = [srcb_bcast](std::uint32_t clr, std::uint32_t addr_mod)
            { return TT_OP_ELWADD(clr, p_elwise::DISABLE_ACCUM, srcb_bcast, addr_mod, 0); };

            ckernel_template temp(outer, inner, elwadd_bcast_instr(p_elwise::CLR_NONE, ADDR_MOD_0));
            if constexpr (BROADCAST_TYPE != BroadcastType::SCALAR)
            {
                temp.set_last_inner_loop_instr(elwadd_bcast_instr(p_elwise::CLR_SRCB_VLD, last_inner_addr_mod));
            }
            temp.set_last_outer_loop_instr(elwadd_bcast_instr(p_elwise::CLR_SRCAB_VLD, ADDR_MOD_0));
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
        else
        {
            constexpr std::uint32_t bcast_row = (BROADCAST_TYPE != BroadcastType::COL) ? 1U : 0U;
            constexpr std::uint32_t bcast_col = (BROADCAST_TYPE != BroadcastType::ROW) ? 1U : 0U;
            const auto movb2d                 = [bcast_col, bcast_row](std::uint8_t addr_mod)
            { return TT_OP_MOVB2D(0, 0, addr_mod, FPU_MOV_ROWS, bcast_col, bcast_row); }; // dst_addr += 1 enables row broadcast

            ckernel_template temp(outer, inner, movb2d(ADDR_MOD_0));
            temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
            if constexpr (BROADCAST_TYPE == BroadcastType::SCALAR)
            {
                temp.set_last_outer_loop_instr(movb2d(ADDR_MOD_1));
            }
            else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
            {
                temp.set_last_inner_loop_instr(movb2d(ADDR_MOD_1));
            }
            temp.program_bank0_sw_cntl(instrn_buffer);
        }
    }
}

/**
 * @brief Init unary-broadcast math: addrmods, mop configuration, reset counters.
 *
 * @tparam BROADCAST_TYPE: Scalar, row, or column broadcast, values = <COL/ROW/SCALAR>
 * @tparam EN_32BIT_DEST: True if the Dest register is in 32-bit mode, values = <true/false>
 * @tparam unpack_to_dest: UNP path wrote to dest
 * @param tensor_shape: Passed to addrmod / MOP setup
 * @note On the unpack thread, pair with @ref _llk_unpack_unary_broadcast_operands_init_ (T0) with matching BROADCAST_TYPE/unpack_to_dest.
 * @note @ref _llk_math_eltwise_unary_broadcast_ runs the configured op with matching template args.
 */
template <BroadcastType BROADCAST_TYPE, bool EN_32BIT_DEST, bool unpack_to_dest>
inline void _llk_math_eltwise_unary_broadcast_init_(const TensorShape tensor_shape)
{
    LLK_ASSERT(
        tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM && tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM,
        "Unary broadcast currently only supports 32x32 tiles (face_r_dim=16, 2x2 faces)");

    _llk_math_eltwise_unary_broadcast_addrmod_<BROADCAST_TYPE, unpack_to_dest>(tensor_shape);
    _llk_math_eltwise_unary_broadcast_mop_config_<BROADCAST_TYPE, EN_32BIT_DEST, unpack_to_dest>(tensor_shape);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Run one tile of unary broadcast math: set dest write addr
 *
 * @tparam unpack_to_dest: When true, UNP_A unpacks to the Dest register
 * @param tile_idx: Destination tile index within current dest bank (SyncHalf)
 * @note Call @ref _llk_math_eltwise_unary_broadcast_init_ with matching template args before this function.
 */
template <bool unpack_to_dest>
inline void _llk_math_eltwise_unary_broadcast_(const std::uint32_t tile_idx)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);

    if constexpr (unpack_to_dest)
    {
        // Wait condition SRCB_VLD is required as MOVD2B doesn't automatically wait
        // for SrcB[MatrixUnit.SrcBBank].AllowedClient == SrcClient::MatrixUnit. MATH drains the
        // preceding math instructions so their source-bank release has landed before SRCB_VLD tests it.
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH, p_stall::WAIT_SFPU, p_stall::SRCB_VLD); // TEN-4367 - SrcB sync workaround
    }

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}
