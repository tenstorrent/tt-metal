// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief MOP configuration for unpacking the broadcast source rows of one tile directly into DEST (UNP_DEST).
 *
 * Only the rows the math broadcast reads are unpacked: SCALAR = face 0 row 0, ROW = row 0 of faces 0 and 1,
 * COL = faces 0 and 2. One run places one tile.
 *
 * @tparam BROADCAST_TYPE: Broadcast type, values = <COL/ROW/SCALAR>
 * @param buf_desc_id: Buffer descriptor of the L1 operand, unpack TRISC partition [0,16)
 */
template <BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_unary_broadcast_operands_to_dest_mop_config_(const std::uint32_t buf_desc_id)
{
    static_assert(BROADCAST_TYPE != BroadcastType::NONE, "Broadcast type cannot be NONE for this operation");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    if constexpr (BROADCAST_TYPE == BroadcastType::SCALAR)
    {
        ckernel_template temp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
    {
        ckernel_template temp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/),
            TT_OP_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 1 /*Dst_Face_Idx*/, 1 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else // BroadcastType::COL
    {
        ckernel_template temp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            TT_OP_UNPACR_DEST_FACE(0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/),
            TT_OP_UNPACR_DEST_FACE(2 /*Dst_Face_Idx*/, 2 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/));
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes the unpacker to unpack the broadcast source of a single operand directly into DEST.
 *
 * Unpack-to-dest counterpart of @ref _llk_unpack_unary_broadcast_operands_init_ (SrcB). Tile calls are
 * bracketed by the section handshake or by dest-dvalid exactly as for @ref _llk_unpack_unary_operand_to_dest_init_.
 *
 * @tparam BROADCAST_TYPE: Broadcast type, values = <COL/ROW/SCALAR>
 * @param buf_desc_id: Buffer descriptor of the L1 operand, unpack TRISC partition [0,16) (see llk_bfd_alloc.h)
 * @note Transpose is forced off for both unpacker engines: UNP_DEST does not support it. 32x32 tiles only.
 * @note UNP_DEST does not widen or narrow: the operand width must match the DEST width (32-bit on a 32-bit DEST, 16-bit on a 16-bit one).
 * @note Math pair (T1): @ref _llk_math_eltwise_unary_broadcast_init_ <BROADCAST_TYPE, EN_32BIT_DEST, true> after
 *       @ref _configure_mov_ops_explicit_alu_data_format_state_ <EN_32BIT_DEST>, then @ref _llk_math_eltwise_unary_broadcast_ <true>
 *       per tile, which reads the rows placed here and writes the broadcast tile back in place. Its MOVD2B stalls on
 *       SRCB_VLD, so this thread sets one @ref _llk_unpack_set_srcB_dummy_valid_ per tile.
 * @note @ref _llk_unpack_unary_broadcast_operands_to_dest_tile_ is the matching execute call on this thread.
 */
template <BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_unary_broadcast_operands_to_dest_init_(const std::uint32_t buf_desc_id)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0 /*TRANSPOSE_EN forced false for UNP_DEST*/);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_unary_broadcast_operands_to_dest_mop_config_<BROADCAST_TYPE>(buf_desc_id);
}

/**
 * @brief Unpacks the broadcast source rows of one tile into DEST tile dst_tile_idx of the current section.
 *        No synchronization.
 *
 * @param l1_tile_idx: Index into the L1 buffer of the tile
 * @param dst_tile_idx: DEST tile index, relative to the current section base. Must be below the section's tile
 *        capacity for the sync mode and DEST width, see @ref get_dest_max_tiles. The math broadcast and the pack
 *        thread address the tile at the same index.
 * @note Call @ref _llk_unpack_unary_broadcast_operands_to_dest_init_ before this function.
 */
inline void _llk_unpack_unary_broadcast_operands_to_dest_tile_(const std::uint32_t l1_tile_idx, const std::uint32_t dst_tile_idx)
{
    // UNP_DEST is driven off the UNP_A bank's counters. The DEST bank is part of the tile index
    // the face / row offsets are immediates of the MOP instructions.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_dest_bank_tile_offset_() + dst_tile_idx);
    ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
