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
 * @file llk_unpack_unary_broadcast_operands_to_dest.h
 * @brief Unpack of the source rows/faces of a scalar, row, or column broadcast straight into DEST (UNP_DEST).
 *
 * Unpack-to-dest counterpart of the SrcB family in llk_unpack_unary_broadcast_operands.h, for a 32-bit DEST. The
 * unpacker writes only the rows/faces the broadcast reads; @ref _llk_math_eltwise_unary_broadcast_ with
 * unpack_to_dest=true then broadcasts them in place. UNP_DEST is driven through UNPACKER0's config and counters.
 */

/**
 * @brief MOP for one tile: UNPACR_DEST of F0 row 0 (SCALAR), F0/F1 row 0 (ROW), or faces F0/F2 (COL). No dvalid.
 *
 * @tparam BROADCAST_TYPE: Broadcast type, values = <COL/ROW/SCALAR>
 * @param buf_desc_id: Buffer descriptor id, allocated from the unpack TRISC partition at op-init time (llk_bfd_alloc.h)
 */
template <BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_unary_broadcast_operands_to_dest_mop_config_(const std::uint32_t buf_desc_id)
{
    static_assert(BROADCAST_TYPE != BroadcastType::NONE, "Broadcast type cannot be NONE for this operation");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    if constexpr (BROADCAST_TYPE == BroadcastType::SCALAR)
    {
        const std::uint32_t unpack_row_instrn =
            TT_OP_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_row_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
    {
        const std::uint32_t unpack_f0_row_instrn =
            TT_OP_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
        const std::uint32_t unpack_f1_row_instrn =
            TT_OP_UNPACR_DEST_ROW(0 /*Dst_Row_Idx*/, 0 /*Src_Row_Idx*/, 1 /*Dst_Face_Idx*/, 1 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_f0_row_instrn, unpack_f1_row_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else // BroadcastType::COL
    {
        const std::uint32_t unpack_f0_instrn = TT_OP_UNPACR_DEST_FACE(0 /*Dst_Face_Idx*/, 0 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
        const std::uint32_t unpack_f2_instrn = TT_OP_UNPACR_DEST_FACE(2 /*Dst_Face_Idx*/, 2 /*Src_Face_Idx*/, 0, 0, buf_desc_id, 0 /*SetDatValid*/);
        ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_f0_instrn, unpack_f2_instrn);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
}

/**
 * @brief Initializes the unpacker to unpack unary broadcast operands directly into DEST.
 *
 * Per-op init: programs the transpose config and the MOP only. It does not touch the DEST bank tracking, so it is safe
 * to call inside a tile loop; the once-per-program bank reset is @ref _llk_unpack_dest_init_.
 *
 * @tparam BROADCAST_TYPE: Broadcast type, values = <COL/ROW/SCALAR>
 * @param buf_desc_id: Buffer descriptor id, allocated from the unpack TRISC partition at op-init time (llk_bfd_alloc.h)
 * @note Math thread (T1): @ref _llk_math_eltwise_unary_broadcast_init_ with unpack_to_dest=true and a 32-bit DEST. Its
 *       MOVD2B reads stall on SrcB dvalid, so this thread sets one @ref _llk_unpack_set_srcB_dummy_valid_ per tile,
 *       ordered after the section handoff that lets math start.
 * @note Synchronization is the caller's, as for @ref _llk_unpack_unary_operand_to_dest_init_: the semaphore section
 *       handshake (tt-metal), or dest-dvalid with @ref _llk_unpack_dest_dvalid_section_done_ (tt-llk tests).
 */
template <BroadcastType BROADCAST_TYPE>
inline void _llk_unpack_unary_broadcast_operands_to_dest_init_(const std::uint32_t buf_desc_id)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0 /*TRANSPOSE_EN forced false for UNP_DEST*/);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_unary_broadcast_operands_to_dest_mop_config_<BROADCAST_TYPE>(buf_desc_id);
}

/**
 * @brief Unpacks the broadcast source of one tile directly into DEST at dst_tile_idx of the current section. No
 *        synchronization.
 *
 * @param l1_tile_idx: Index into the L1 buffer of the tile
 * @param dst_tile_idx: DEST tile index, relative to the current section base. Must be below the section's tile capacity
 *        (@ref get_dest_max_tiles); math broadcasts and pack reads the tile at the same index.
 * @note Call @ref _llk_unpack_unary_broadcast_operands_to_dest_init_ before this function.
 */
inline void _llk_unpack_unary_broadcast_operands_to_dest_tile_(const std::uint32_t l1_tile_idx, const std::uint32_t dst_tile_idx)
{
    // UNP_DEST is driven off the UNP_A bank's counters; the DEST bank is part of the tile index.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_dest_bank_tile_offset_() + dst_tile_idx);
    ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
