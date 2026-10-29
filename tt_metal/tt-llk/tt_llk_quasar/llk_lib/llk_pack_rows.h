// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_buf_desc.h"
#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_memory_checks.h"
#include "llk_pack_common.h"
#include "tensix_types.h"

/**
 * @brief Builds the replay buffer and MOP for packing rows via Packer 0.
 *
 * The MOP issues one PACR0_FACE_INC per full face, which packs the whole face and moves the DEST and L1 face counters
 * forward. The remaining num_rows % 16 rows come from the replay buffer, which holds 15 PACR0_ROW_INC, each moving
 * the DEST and L1 row counters forward by one row and leaving the face counters alone.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table; allocated from the pack TRISC partition [16,24) at op-init time (see llk_bfd_alloc.h)
 * @param num_rows: Number of 16-datum rows to pack, values = [1, 64]
 */
inline void _llk_pack_rows_mop_config_(const std::uint8_t buf_desc_id, const std::uint32_t num_rows)
{
    constexpr std::uint32_t replay_buf_len = ckernel::FACE_R_DIM - 1;

    ckernel::load_replay_buf<0, replay_buf_len>(
        [buf_desc_id]
        {
            for (std::uint32_t row = 0; row < replay_buf_len; row++)
            {
                TT_PACR0_ROW_INC(
                    1 /*Dst_Row_Idx_Inc*/,
                    1 /*Src_Row_Idx_Inc*/,
                    0 /*Dst_Face_Idx_Inc*/,
                    0 /*Src_Face_Idx_Inc*/,
                    0 /*Dst_Tile_Offset_Idx_Inc*/,
                    0 /*Src_Tile_Offset_Idx_Inc*/,
                    buf_desc_id,
                    0 /*ClrDatValid*/);
            }
        });

    const std::uint32_t num_full_faces = num_rows / ckernel::FACE_R_DIM;
    const std::uint32_t num_tail_rows  = num_rows % ckernel::FACE_R_DIM;
    const std::uint32_t pack_face      = TT_OP_PACR0_FACE_INC(
        1 /*Dst_Face_Idx_Inc*/, 1 /*Src_Face_Idx_Inc*/, 0 /*Dst_Tile_Offset_Idx_Inc*/, 0 /*Src_Tile_Offset_Idx_Inc*/, buf_desc_id, 0 /*ClrDatValid*/);
    const std::uint32_t replay_tail =
        TT_OP_REPLAY(0 /*start_idx*/, num_tail_rows /*len*/, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);

    if (num_full_faces == 0)
    {
        ckernel::ckernel_template temp(1 /*outer*/, 1 /*inner*/, replay_tail);
        temp.program_bank0_sw_cntl(ckernel::instrn_buffer);
    }
    else
    {
        ckernel::ckernel_template temp(1 /*outer*/, num_full_faces /*inner*/, pack_face);
        if (num_tail_rows != 0)
        {
            temp.set_end_op(replay_tail);
        }
        temp.program_bank0_sw_cntl(ckernel::instrn_buffer);
    }
}

/**
 * @brief Initializes the packer for packing the first num_rows rows of a DEST tile to an L1 address via Packer 0.
 *
 * Each row is 16 datums. Rows are read in DEST order (face 0 rows 0-15, then face 1, ...) and written contiguously.
 * The buffer descriptor must describe a 32x32 tile in a non-MX output format; its L1 base is replaced on every
 * @ref _llk_pack_rows_ call, so it must not be shared with another pack op.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table; allocated from the pack TRISC partition [16,24) at op-init time (see llk_bfd_alloc.h)
 * @param num_rows: Number of 16-datum rows to pack, values = [1, 64]
 * @note @ref _llk_pack_rows_ is the matching execute call on this thread, and @ref _llk_pack_rows_uninit_ ends the op.
 */
inline void _llk_pack_rows_init_(const std::uint8_t buf_desc_id, const std::uint32_t num_rows)
{
    // num_rows == 0 would build a zero-length REPLAY, which hangs the thread.
    LLK_ASSERT(num_rows >= 1 && num_rows <= ckernel::trisc::NUM_FACES * ckernel::FACE_R_DIM, "num_rows must be between 1 and 64");

    _llk_pack_rows_mop_config_(buf_desc_id, num_rows);
}

/**
 * @brief Packs the rows set up by @ref _llk_pack_rows_init_ from a DEST tile to an L1 address via Packer 0.
 *
 * Bypasses the DFB write pointer: only the num_rows * 16 datums starting at l1_address are written.
 *
 * @param buf_desc_id: The buffer descriptor ID passed to @ref _llk_pack_rows_init_
 * @param dest_tile_idx: The tile index into the math destination register to pack from
 * @param l1_address: L1 address to pack to, in 16B units
 * @note Call @ref _llk_pack_rows_init_ before this function.
 */
inline void _llk_pack_rows_(const std::uint8_t buf_desc_id, const std::uint32_t dest_tile_idx, const std::uint32_t l1_address)
{
    LLK_ASSERT(is_valid_L1_address(l1_address), "L1 address must be in valid L1 memory region");
    ckernel::trisc::_set_buf_desc_l1_base_(buf_desc_id, l1_address);

    // Reset the DEST and L1 counters: the previous call's rows, or a prior op such as pack untilize, may have left them
    // part way through a tile. Then select the DEST tile to read.
    TTI_SET_SRC_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_FACE_ROW_SEL, ckernel::p_pacr::PACK0, 0 /*Value*/);
    TTI_SET_DST_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_FACE_ROW_SEL, ckernel::p_pacr::PACK0, 0 /*Value*/);
    TT_SET_SRC_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_SEL, ckernel::p_pacr::PACK0, dest_tile_idx);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(ckernel::instrn_buffer);
}

/**
 * @brief Uninitializes pack rows: clears the packer tile/face/row counters.
 *
 * The row pack leaves the face and row counters part way through a tile, and setting only the tile or face counter
 * does not clear the counters below it. Strided pack untilize sets only the tile counters, so a stale L1 face
 * counter would shift its writes by a whole tile.
 *
 * @note Pairs with @ref _llk_pack_rows_init_.
 */
inline void _llk_pack_rows_uninit_()
{
    TTI_SET_SRC_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_FACE_ROW_SEL, ckernel::p_pacr::PACK0, 0 /*Value*/);
    TTI_SET_DST_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_FACE_ROW_SEL, ckernel::p_pacr::PACK0, 0 /*Value*/);
}
