// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @file llk_unpack_unary_broadcast_operands.h
 * @brief UNP replay / MOP setup and per-tile run for unary eltwise with scalar, row, or column broadcast, unpacking to SrcB.
 * The unpack-to-dest counterpart is llk_unpack_unary_broadcast_operands_to_dest.h.
 */

/**
 * @brief Builds the MOP for unpacking unary broadcast operands into SrcB.
 *
 * @tparam UNP_SEL: Unpacker select; must be UNP_B, values = <p_unpacr::UNP_B>
 * @tparam BROADCAST_TYPE: Broadcast type, values = <COL/ROW/SCALAR>
 * @tparam EN_32BIT_DEST: True if the Dest register is in 32-bit mode, values = <true/false>
 * @param buf_desc_id: Buffer descriptor for the UNPACR source.
 * @param num_tiles: Outer MOP loop count (tiles to unpack from L1).
 */
template <std::uint32_t UNP_SEL, BroadcastType BROADCAST_TYPE, bool EN_32BIT_DEST>
inline void _llk_unpack_unary_broadcast_operands_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
{
    static_assert(UNP_SEL == p_unpacr::UNP_B, "UNP_SEL must be p_unpacr::UNP_B - movA2D broadcast is not working on Quasar");
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");

    const std::uint32_t MOP_OUTER_LOOP            = num_tiles;
    constexpr std::uint32_t MOP_INNER_LOOP        = 1;
    constexpr static std::uint32_t replay_buf_len = BROADCAST_TYPE == BroadcastType::SCALAR ? 1u : 4u;

    if constexpr (BROADCAST_TYPE == BroadcastType::SCALAR)
    {
        load_replay_buf<0, replay_buf_len>(
            [buf_desc_id]
            { TT_UNPACR1_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/); });
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
    {
        load_replay_buf<0, replay_buf_len>(
            [buf_desc_id]
            {
                // Match face order in llk_unpack_binary_broadcast_operands.h (UNP_B / SrcB path)
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 1 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 1 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
            });
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
    {
        load_replay_buf<0, replay_buf_len>(
            [buf_desc_id]
            {
                // Match face order in llk_unpack_binary_broadcast_operands.h (UNP_B / SrcB path)
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
            });
    }

    const std::uint32_t inc_src_tile_instrn = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 1);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), inc_src_tile_instrn);
    if constexpr (EN_32BIT_DEST)
    {
        // 32-bit dest uses ELWADD, which requires dvalid from both SrcA and SrcB.
        // Data is unpacked to SrcB, so zero the unused engine (SrcA) and set its dvalid.
        const std::uint32_t clr_unused_unpacr_engine = nop_insn_for_unused_unpacker_engine<UNP_SEL>();
        temp.set_start_op(clr_unused_unpacr_engine);
    }
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker for unary operations with broadcasts, unpacking into SrcB.
 *
 * @tparam UNP_SEL: Unpacker resource; must be UNP_B, values = <p_unpacr::UNP_B>
 * @tparam BROADCAST_TYPE: Broadcast type, values = <COL/ROW/SCALAR>
 * @tparam EN_32BIT_DEST: True if the Dest register is in 32-bit mode, values = <true/false>
 * @param buf_desc_id: Buffer descriptor for the UNPACR source.
 * @param num_tiles: Number of tiles in the outer unpack loop.
 * @note On the math thread, pair with @ref _llk_math_eltwise_unary_broadcast_init_ (T1) with matching BROADCAST_TYPE/EN_32BIT_DEST
 *       and unpack_to_dest=false. The unpack-to-dest counterpart is @ref _llk_unpack_unary_broadcast_operands_to_dest_init_.
 * @note @ref _llk_unpack_unary_broadcast_operands_ is the matching execute call on this thread.
 */
template <std::uint32_t UNP_SEL, BroadcastType BROADCAST_TYPE, bool EN_32BIT_DEST>
inline void _llk_unpack_unary_broadcast_operands_init_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_unary_broadcast_operands_mop_config_<UNP_SEL, BROADCAST_TYPE, EN_32BIT_DEST>(buf_desc_id, num_tiles);
}

/**
 * @brief Runs the bank0 unpack MOP for one invocation (sets tile/face indices then executes the replay).
 *
 * @tparam UNP_SEL: Unpacker resource; must be UNP_B, values = <p_unpacr::UNP_B>
 * @param start_l1_tile_idx: Starting source tile index for face/row counters.
 * @note Call @ref _llk_unpack_unary_broadcast_operands_init_ with matching template args before this function.
 */
template <std::uint32_t UNP_SEL>
inline void _llk_unpack_unary_broadcast_operands_(const std::uint32_t start_l1_tile_idx)
{
    static_assert(UNP_SEL == p_unpacr::UNP_B, "UNP_SEL must be p_unpacr::UNP_B - movA2D broadcast is not working on Quasar");

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, start_l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, UNP_SEL, 0);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
