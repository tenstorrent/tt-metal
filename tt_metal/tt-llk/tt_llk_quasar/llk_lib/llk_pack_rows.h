// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_buf_desc.h"
#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_pack_common.h"
#include "tensix_types.h"

using namespace ckernel;

/**
 * @brief Builds the replay buffer and MOP for packing rows via Packer 0.
 *
 * The replay buffer holds one face: 16 PACR0_ROW_INC, each moving the DEST and L1 row counters forward by one row.
 * The row counters wrap to 0 at the buffer descriptor's y_dim without carrying into the face counters, so the 16th
 * PACR also moves both face counters forward. The MOP replays the whole face once per full face, then replays only
 * the first num_rows % 16 entries, which leave the face counters alone.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table; allocated from the pack TRISC partition [16,24) at op-init time (see llk_bfd_alloc.h)
 * @param num_rows: Number of 16-datum rows to pack, values = [1, 64]
 */
inline void _llk_pack_rows_mop_config_(const std::uint8_t buf_desc_id, const std::uint32_t num_rows)
{
    constexpr std::uint32_t replay_buf_len = FACE_R_DIM;

    load_replay_buf<0, replay_buf_len>(
        [buf_desc_id]
        {
            for (std::uint32_t row = 0; row < replay_buf_len - 1; row++)
            {
                TT_PACR0_ROW_INC(
                    1 /*Dst_Row_Idx_Inc*/, 1 /*Src_Row_Idx_Inc*/, 0 /*Dst_Face_Idx_Inc*/, 0 /*Src_Face_Idx_Inc*/, 0, 0, buf_desc_id, 0 /*ClrDatValid*/);
            }
            TT_PACR0_ROW_INC(
                1 /*Dst_Row_Idx_Inc*/, 1 /*Src_Row_Idx_Inc*/, 1 /*Dst_Face_Idx_Inc*/, 1 /*Src_Face_Idx_Inc*/, 0, 0, buf_desc_id, 0 /*ClrDatValid*/);
        });

    const std::uint32_t num_full_faces = num_rows / FACE_R_DIM;
    const std::uint32_t num_tail_rows  = num_rows % FACE_R_DIM;
    const std::uint32_t replay_face    = TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
    const std::uint32_t replay_tail    = TT_OP_REPLAY(0, num_tail_rows, 0, 0, 0, 0);

    if (num_full_faces == 0)
    {
        ckernel_template temp(1 /*outer*/, 1 /*inner*/, replay_tail);
        temp.program_bank0_sw_cntl(instrn_buffer);
    }
    else
    {
        ckernel_template temp(1 /*outer*/, num_full_faces /*inner*/, replay_face);
        if (num_tail_rows != 0)
        {
            temp.set_end_op(replay_tail);
        }
        temp.program_bank0_sw_cntl(instrn_buffer);
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
    LLK_ASSERT(num_rows >= 1 && num_rows <= NUM_FACES * FACE_R_DIM, "num_rows must be between 1 and 64");
    // MX formats pack two rows per exponent block and write a separate exponent section, so they would write outside
    // the requested rows.
    const DataFormat l1_format = ckernel::trisc::_get_buf_desc_l1_format_(buf_desc_id);
    LLK_ASSERT(
        l1_format != DataFormat::MxFp8R && l1_format != DataFormat::MxFp8P && l1_format != DataFormat::MxFp6R && l1_format != DataFormat::MxFp6P &&
            l1_format != DataFormat::MxFp4 && l1_format != DataFormat::MxInt8 && l1_format != DataFormat::MxInt4 && l1_format != DataFormat::MxInt2,
        "pack rows does not support MX output formats");

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
    ckernel::trisc::_set_buf_desc_l1_base_(buf_desc_id, l1_address);

    // Reset the DEST and L1 counters: the previous call's rows, or a prior op such as pack untilize, may have left them
    // part way through a tile. Then select the DEST tile to read.
    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, p_pacr::PACK0, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, p_pacr::PACK0, 0);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, dest_tile_idx);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
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
    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, p_pacr::PACK0, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, p_pacr::PACK0, 0);
}
