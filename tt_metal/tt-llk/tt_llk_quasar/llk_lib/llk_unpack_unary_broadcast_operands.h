// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_common.h"
using namespace ckernel;

/**
 * @file llk_unpack_unary_broadcast_operands.h
 * @brief UNP replay / MOP setup and per-tile run for unary eltwise with scalar, row, or column broadcast.
 */

/**
 * @brief MOP config for unpack of unary broadcast operands.
 * @tparam UNP_SEL Unpacker select; must be UNP_B unless unpack_to_dest (then UNP_A).
 * @tparam BROADCAST_TYPE SCALAR, COL, or ROW (not NONE).
 * @tparam unpack_to_dest When true, unpack targets math dest (UNP_A); otherwise srcB (UNP_B).
 * @tparam is_fp32_dest_acc_en Float32 dest accumulation enable. Must be false when unpack_to_dest is true
 *         until that path is supported (enforced by static_assert below).
 * @param buf_desc_id Buffer descriptor for UNPACR source.
 * @param num_tiles Outer MOP loop count (tiles to unpack from L1).
 */
template <std::uint32_t UNP_SEL, BroadcastType BROADCAST_TYPE, bool unpack_to_dest = false, bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_unary_broadcast_operands_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
{
    static_assert(
        unpack_to_dest || (UNP_SEL == p_unpacr::UNP_B),
        "UNP_SEL must be p_unpacr::UNP_B when unpack_to_dest is false - movA2D broadcast is not working on Quasar");
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");
    static_assert(!(unpack_to_dest && is_fp32_dest_acc_en), "Unary broadcast: unpack_to_dest with Float32 dest accumulation is not supported yet");

    const std::uint32_t MOP_OUTER_LOOP            = num_tiles;
    constexpr std::uint32_t MOP_INNER_LOOP        = 1;
    constexpr static std::uint32_t replay_buf_len = BROADCAST_TYPE == BroadcastType::SCALAR ? 1u : (unpack_to_dest ? 2u : 4u);

    if constexpr (BROADCAST_TYPE == BroadcastType::SCALAR)
    {
        load_replay_buf<0, replay_buf_len>(
            [buf_desc_id]
            {
                if constexpr (unpack_to_dest)
                {
                    TT_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                }
                else
                {
                    TT_UNPACR1_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                }
            });
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
    {
        load_replay_buf<0, replay_buf_len>(
            [buf_desc_id]
            {
                if constexpr (unpack_to_dest)
                {
                    TT_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
                    TT_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 1 /*Dst_Face_Idx*/, 1 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                }
                else
                {
                    // Match face order in llk_unpack_binary_broadcast_operands.h (UNP_B / SrcB path)
                    TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                    TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 1 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                    TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                    TT_UNPACR1_FACE(0 /*Dst Face Idx*/, 1 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
                }
            });
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
    {
        load_replay_buf<0, replay_buf_len>(
            [buf_desc_id]
            {
                if constexpr (unpack_to_dest)
                {
                    TT_UNPACR_DEST_FACE(0 /*Dst Face Idx*/, 0 /*Src Face Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
                    TT_UNPACR_DEST_FACE(2 /*Dst Face Idx*/, 2 /*Src Face Idx*/, 0, 0, buf_desc_id, 1 /*SetDatValid*/);
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
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for unpack of unary operations with broadcasts.
 * @tparam UNP_SEL Unpacker resource; must be UNP_B unless unpack_to_dest.
 * @tparam BROADCAST_TYPE SCALAR, COL, or ROW.
 * @tparam unpack_to_dest Route unpack to dest (UNP_A) vs srcB (UNP_B).
 * @tparam is_fp32_dest_acc_en Forwarded to mop_config; must be false when unpack_to_dest is true.
 * @param buf_desc_id Buffer descriptor for UNPACR source.
 * @param num_tiles Number of tiles in the outer unpack loop.
 */
template <std::uint32_t UNP_SEL, BroadcastType BROADCAST_TYPE, bool unpack_to_dest = false, bool is_fp32_dest_acc_en = false>
inline void _llk_unpack_unary_broadcast_operands_init_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
{
    _llk_unpack_unary_broadcast_operands_mop_config_<UNP_SEL, BROADCAST_TYPE, unpack_to_dest, is_fp32_dest_acc_en>(buf_desc_id, num_tiles);
}

/**
 * @brief Runs bank0 unpack MOP for one invocation (sets tile/face indices then executes replay).
 * @tparam UNP_SEL Logical unpack select for static checks; counter uses UNP_A when unpack_to_dest.
 * @tparam unpack_to_dest Use UNP_A dest path vs UNP_B srcB path.
 * @param start_l1_tile_idx Starting source tile index for face/row counters.
 */
template <std::uint32_t UNP_SEL, bool unpack_to_dest = false>
inline void _llk_unpack_unary_broadcast_operands_(const std::uint32_t start_l1_tile_idx)
{
    static_assert(
        unpack_to_dest || (UNP_SEL == p_unpacr::UNP_B),
        "UNP_SEL must be p_unpacr::UNP_B when unpack_to_dest is false - movA2D broadcast is not working on Quasar");

    constexpr std::uint32_t counter_unp_sel = unpack_to_dest ? p_unpacr::UNP_A : UNP_SEL;
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, counter_unp_sel, start_l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, counter_unp_sel, 0);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
