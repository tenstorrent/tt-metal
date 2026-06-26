// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_sync.h"
#include "llk_unpack_common.h"

using namespace ckernel;

/**
 * @brief MOP configuration for unpacking a single operand directly into the math DEST register.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer
 *        descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for a single operand
 * @note UNP_DEST places consecutive faces at consecutive DEST positions (Dst_Tile_Idx_Inc = 1); no math is
 *       involved, so no per-face dvalid is set.
 */
inline void _llk_unpack_to_dest_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
{
    const std::uint32_t MOP_OUTER_LOOP     = num_tiles;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    const std::uint32_t unpack_tile_instrn = TT_OP_UNPACR_DEST_TILE_INC(1 /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker to unpack a single operand directly into the math DEST register.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer
 *        descriptor table, values = 0 - 16
 * @param num_tiles: number of tiles to unpack at a time for a single operand; default 1 tile of 32x32
 * @note Unpack owns the DEST section base in this path: it is the DEST producer (UNP_DEST), so it programs the
 *       per-TRISC section base itself rather than letting the math middleman set it on its behalf. unpack::TRISC_ID
 *       selects the same SEC slot the UNP_DEST client reads; the per-tile call flips it in SyncHalf.
 * @note On the math thread (T1) pair with @ref _llk_math_eltwise_unary_datacopy_init_; @ref _llk_unpack_to_dest_ is
 *       the matching execute call on this thread.
 */
inline void _llk_unpack_to_dest_init_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
{
    // Establish the initial bank-0 base here; the per-tile call flips it in SyncHalf.
    ckernel::trisc::_reset_dest_register_offset_();
    ckernel::trisc::_set_dest_section_base_<ckernel::unpack::TRISC_ID>(ckernel::trisc::_get_dest_buffer_base_());

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0 /*TRANSPOSE_EN forced false for UNP_DEST*/);
    _llk_unpack_to_dest_mop_config_(buf_desc_id, num_tiles);
}

/**
 * @brief Unpacks a single operand directly into the math DEST register.
 * @tparam DEST_SYNC_MODE: In SyncHalf, flips the DEST section base to the other bank after each tile, values = <SyncFull/SyncHalf>
 * @param l1_tile_idx: Index into the L1 buffer for a tile.
 * @note The math thread is the middleman with two single-counting semaphores (max = N each). Without an extra wait on
 *       MATH_PACK, unpack could race 2N iterations ahead of pack and overwrite a bank pack has not read yet; waiting on
 *       both keeps unpack within N iterations of pack.
 * @note Call @ref _llk_unpack_to_dest_init_ before this function.
 */
template <ckernel::DstSync DEST_SYNC_MODE = ckernel::DstSync::SyncFull>
inline void _llk_unpack_to_dest_(const std::uint32_t l1_tile_idx)
{
    _llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::MATH_PACK, semaphore::UNPACK_MATH);

    // UNP_DEST is driven off the UNP_A bank's counters.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

    // Drain UNPACK0 before posting "filled" so the post does not race the writes math reads.
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);

    // Unpack owns the DEST section base, so it flips to the other bank for the next iteration.
    if constexpr (DEST_SYNC_MODE == ckernel::DstSync::SyncHalf)
    {
        _llk_sync_advance_dest_section_<ckernel::unpack::TRISC_ID, true /*EN_32BIT_DEST*/, p_stall::UNPACK0>();
    }
}
