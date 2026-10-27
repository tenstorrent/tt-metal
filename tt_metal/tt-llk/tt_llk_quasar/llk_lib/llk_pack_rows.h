// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_pack_common.h"
#include "tensix_types.h"

using namespace ckernel;

/**
 * @brief Points a buffer descriptor at a new L1 base from the pack instruction stream.
 *
 * A PACR copies its descriptor as it leaves the issue buffer and RMWCIB (issued by cfg_rmw) writes as it leaves, so
 * PACRs issued earlier keep the old base and PACRs issued later see the new one, without a stall.
 *
 * @param buf_desc_id: The buffer descriptor ID to retarget
 * @param l1_address: New L1 base address, in 16B units
 */
inline void _llk_pack_rows_set_l1_address_(const std::uint8_t buf_desc_id, const std::uint32_t l1_address)
{
    // bd_table entries are buffer_descriptor_u, back to back from the table base.
    const std::uint32_t word0_addr32 = BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_ADDR32 + buf_desc_id * (sizeof(buffer_descriptor_u) / sizeof(std::uint32_t));
    LLK_ASSERT(l1_address <= BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_MASK, "l1_address exceeds the buffer descriptor L1 base field");

    cfg_rmw(word0_addr32, BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_SHAMT, BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_MASK, l1_address);
}

/**
 * @brief Builds the replay buffer and MOP for packing rows via Packer 0.
 *
 * The replay buffer holds one face: 16 PACR0_ROW_INC, each moving the DEST and L1 row counters forward by one row.
 * The row counters wrap at the buffer descriptor's y_dim without carrying into the face counters, so the 16th PACR
 * also moves both face counters forward. The MOP replays the whole face once per full face, then replays only the
 * first num_rows % 16 entries, which leave the face counters alone.
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
 * The buffer descriptor must describe a 32x32 tile in the output format; its L1 base is replaced on every
 * @ref _llk_pack_rows_ call, so it must not be shared with another pack op.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table; allocated from the pack TRISC partition [16,24) at op-init time (see llk_bfd_alloc.h)
 * @param num_rows: Number of 16-datum rows to pack, values = [1, 64]
 * @note @ref _llk_pack_rows_ is the matching execute call on this thread.
 */
inline void _llk_pack_rows_init_(const std::uint8_t buf_desc_id, const std::uint32_t num_rows)
{
    LLK_ASSERT(num_rows >= 1 && num_rows <= NUM_FACES * FACE_R_DIM, "num_rows must be between 1 and 64");

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
    _llk_pack_rows_set_l1_address_(buf_desc_id, l1_address);

    // Reset the DEST and L1 counters: the previous call's rows left them part way through the tile, then select the
    // DEST tile to read.
    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, p_pacr::PACK0, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_FACE_ROW_SEL, p_pacr::PACK0, 0);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, dest_tile_idx);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Uninitializes pack rows. Empty on Quasar: every pack op's init programs the state it uses.
 *
 * @note Pairs with @ref _llk_pack_rows_init_.
 */
inline void _llk_pack_rows_uninit_()
{
}
