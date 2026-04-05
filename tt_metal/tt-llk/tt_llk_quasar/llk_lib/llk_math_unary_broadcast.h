// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @file llk_math_unary_broadcast.h
 * @brief Math addrmods, MOP, and per-tile run for unary eltwise with scalar, row, or column broadcast.
 */

/**
 * @brief Programs address modifiers for eltwise unary broadcast (MOVB2D / MOVD2B paths).
 * @tparam BROADCAST_TYPE Scalar, row, or column broadcast; must not be NONE.
 * @tparam unpack_to_dest When true, UNP_A wrote to dest; ADDR_MOD_3 used for dest<->srcB moves.
 * @param tile_shape Face geometry (face_r_dim, num_faces) for row-broadcast addrmods.
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_broadcast_addrmod_(const TileShape& tile_shape)
{
    static_assert(BROADCAST_TYPE != BroadcastType::NONE, "Broadcast type cannot be NONE");

    constexpr std::uint16_t row_step    = static_cast<std::uint16_t>(ELTWISE_MATH_ROWS);
    constexpr std::uint8_t srcb_col_inc = (BROADCAST_TYPE == BroadcastType::COL) ? static_cast<std::uint8_t>(ELTWISE_MATH_ROWS) : static_cast<std::uint8_t>(0);

    addr_mod_t {.srcb = {.incr = srcb_col_inc}, .dest = {.incr = row_step}}.set(ADDR_MOD_0);
    addr_mod_t {.srcb = {.clr = 1}, .dest = {.incr = row_step}}.set(ADDR_MOD_1);

    if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
    {
        addr_mod_t {
            .srcb = {.incr = static_cast<std::uint8_t>(tile_shape.face_r_dim)},
            .dest = {.incr = row_step},
        }
            .set(ADDR_MOD_2);
    }

    if constexpr (unpack_to_dest)
    {
        if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
        {
            addr_mod_t {
                .srcb = {.incr = static_cast<std::uint8_t>(tile_shape.face_r_dim)},
                .dest = {.incr = static_cast<std::uint16_t>(tile_shape.face_r_dim)},
            }
                .set(ADDR_MOD_3);
        }
        else
        {
            addr_mod_t {
                .srcb = {.incr = static_cast<std::uint8_t>(ELTWISE_MATH_ROWS)},
                .dest = {.incr = static_cast<std::uint16_t>(ELTWISE_MATH_ROWS)},
            }
                .set(ADDR_MOD_3);
        }
    }
}

/**
 * @brief MOP / replay configuration for MOVB2D unary-broadcast (srcB -> dest).
 * @tparam BROADCAST_TYPE Scalar, row, or column broadcast.
 * @tparam unpack_to_dest When true, unpack filled dest; MOVB2D reads srcB only, so `_llk_math_eltwise_unary_broadcast_`
 *         runs MOVD2B (dest->srcB) first, then programs this MOVB2D MOP. ROW uses the fixed replay below; COL/SCALAR use
 *         the same outer/inner template as the non-unpack_to_dest path with ADDR_MOD_3 from addrmod.
 * @param tile_shape Tile shape for loop counts and row dimensions.
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest = false>
inline void _llk_math_eltwise_unary_broadcast_mop_config_(const TileShape& tile_shape)
{
    static_assert(BROADCAST_TYPE != BroadcastType::NONE, "Broadcast type cannot be NONE");

    if constexpr (BROADCAST_TYPE == BroadcastType::ROW && unpack_to_dest)
    {
        constexpr std::uint32_t replay_buf_start          = 1;
        constexpr std::uint32_t replay_movb2d_instr_count = ELTWISE_MATH_ROWS;
        constexpr std::uint32_t replay_buf_len            = replay_movb2d_instr_count;
        load_replay_buf<replay_buf_start, replay_buf_len>(
            []
            {
                TTI_MOVB2D(0, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, 0, 1);
                TTI_MOVB2D(0, 0, ADDR_MOD_2, p_mov_src_to_dest::MOV_8_ROWS, 0, 1);
                TTI_MOVB2D(0, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, 0, 1);
                TTI_MOVB2D(0, 0, ADDR_MOD_1, p_mov_src_to_dest::MOV_8_ROWS, 0, 1);
                TTI_MOVB2D(0, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, 0, 1);
                TTI_MOVB2D(0, 0, ADDR_MOD_2, p_mov_src_to_dest::MOV_8_ROWS, 0, 1);
                TTI_MOVB2D(0, 0, ADDR_MOD_0, p_mov_src_to_dest::MOV_8_ROWS, 0, 1);
                TTI_MOVB2D(0, 0, ADDR_MOD_2, p_mov_src_to_dest::MOV_8_ROWS, 0, 1);
            });

        ckernel_template temp(1, 1, TT_OP_REPLAY(replay_buf_start, replay_buf_len, 0, 0, 0, 0));
        temp.set_end_op(TT_OP_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, 0, 0));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        const std::uint32_t num_rows = (BROADCAST_TYPE == BroadcastType::SCALAR) ? tile_shape.num_faces * tile_shape.face_r_dim : tile_shape.face_r_dim;
        const std::uint32_t outer    = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1U : static_cast<std::uint32_t>(tile_shape.num_faces);
        const std::uint32_t inner    = num_rows >> math_rows_log2(ELTWISE_MATH_ROWS);

        const std::uint32_t dst_lo     = (BROADCAST_TYPE != BroadcastType::COL) ? 1U : 0U;
        constexpr std::uint32_t bcast0 = (BROADCAST_TYPE == BroadcastType::COL || BROADCAST_TYPE == BroadcastType::SCALAR) ? 1U : 0U;

        const auto movb2d = [bcast0, dst_lo](std::uint8_t am) { return TT_OP_MOVB2D(0, 0, am, p_mov_src_to_dest::MOV_8_ROWS, bcast0, dst_lo); };

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
        else
        {
            temp.set_last_inner_loop_instr(movb2d(ADDR_MOD_0));
            temp.set_last_outer_loop_instr(movb2d(ADDR_MOD_2));
        }

        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief MOP for MOVD2B when unpack wrote to dest (move dest rows back to srcB for MOVB2D).
 * @tparam BROADCAST_TYPE Scalar, row, or column broadcast.
 * @param tile_shape Used for row/col loop counts outside ROW special case.
 */
template <BroadcastType BROADCAST_TYPE>
inline void _llk_math_eltwise_unary_broadcast_d2b_mop_config_(const TileShape& tile_shape)
{
    if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
    {
        const auto f = [](std::uint8_t am, std::uint32_t d32) { return TT_OP_MOVD2B(d32, 0, am, p_movd2b::MOV_1_ROW, 0, 0); };

        constexpr std::uint32_t replay_buf_len = 1;
        load_replay_buf<0, replay_buf_len>([] { TTI_MOVD2B(0, 0, ADDR_MOD_3, p_movd2b::MOV_1_ROW, 0, 0); });

        ckernel_template temp(1, 1, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), f(ADDR_MOD_3, 0));
        temp.set_last_inner_loop_instr(f(ADDR_MOD_3, 1));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        // SCALAR: all faces × rows. COL: column broadcast touches two faces in dest (mirrors unpack UNPACR_DEST_FACE 0 and 2), so row count is 2× face_r_dim.
        const std::uint32_t rows_sel  = (BROADCAST_TYPE == BroadcastType::SCALAR) ? tile_shape.num_faces * tile_shape.face_r_dim : tile_shape.face_r_dim * 2;
        const std::uint32_t inner_d2b = (BROADCAST_TYPE == BroadcastType::SCALAR) ? 1U : (rows_sel >> math_rows_log2(ELTWISE_MATH_ROWS));

        const auto f = [](std::uint8_t am, std::uint32_t d32) { return TT_OP_MOVD2B(d32, 0, am, p_mov_src_to_dest::MOV_8_ROWS, 0, 0); };

        constexpr std::uint32_t replay_buf_len = 1;
        load_replay_buf<0, replay_buf_len>([] { TTI_MOVD2B(0, 0, ADDR_MOD_3, p_mov_src_to_dest::MOV_8_ROWS, 0, 0); });

        ckernel_template temp(1, inner_d2b * 2, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), f(ADDR_MOD_3, 0));
        temp.set_last_inner_loop_instr(f(ADDR_MOD_3, 1));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Init unary-broadcast math: addrmods, MOP when not unpack_to_dest, reset counters.
 * @tparam BROADCAST_TYPE Scalar, row, or column broadcast.
 * @tparam unpack_to_dest UNP path wrote to dest; MOVB2D MOP deferred to per-tile call.
 * @tparam is_fp32_dest_acc_en Same name/position as unpack init for uniform call sites. Must be false when
 *         unpack_to_dest is true (static_assert below).
 * @param tile_shape Passed to addrmod / MOP setup.
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest = false, bool is_fp32_dest_acc_en = false>
inline void _llk_math_eltwise_unary_broadcast_init_(const TileShape& tile_shape)
{
    static_assert(!(unpack_to_dest && is_fp32_dest_acc_en), "Unary broadcast: unpack_to_dest with Float32 dest accumulation is not supported yet");
    _llk_math_eltwise_unary_broadcast_addrmod_<BROADCAST_TYPE, unpack_to_dest>(tile_shape);
    if constexpr (!unpack_to_dest)
    {
        _llk_math_eltwise_unary_broadcast_mop_config_<BROADCAST_TYPE, false>(tile_shape);
    }
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Run one tile of unary broadcast math: set dest write addr, optional D2B then MOVB2D when unpack_to_dest.
 * @tparam BROADCAST_TYPE Scalar, row, or column broadcast.
 * @tparam unpack_to_dest When true, runs D2B then MOVB2D replay for unpack-to-dest workaround.
 * @tparam is_fp32_dest_acc_en Same template args as init; combination unpack_to_dest + true is rejected in init.
 * @param tile_idx Destination tile index within current dest bank (SyncHalf).
 * @param tile_shape Used when unpack_to_dest (D2B / MOVB2D MOP); otherwise ignored.
 */
template <BroadcastType BROADCAST_TYPE, bool unpack_to_dest = false, bool is_fp32_dest_acc_en = false>
inline void _llk_math_eltwise_unary_broadcast_(const std::uint32_t tile_idx, [[maybe_unused]] const TileShape& tile_shape)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);

    if constexpr (unpack_to_dest)
    {
        _llk_math_eltwise_unary_broadcast_d2b_mop_config_<BROADCAST_TYPE>(tile_shape);
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        _reset_counters_<p_setrwc::SET_ABD_F>();
        _llk_math_eltwise_unary_broadcast_mop_config_<BROADCAST_TYPE, true>(tile_shape);
    }

    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
