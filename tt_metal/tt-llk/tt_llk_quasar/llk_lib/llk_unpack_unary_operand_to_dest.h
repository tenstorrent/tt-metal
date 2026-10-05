// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_sync.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief Reset the unpack thread's DEST bank tracking to bank 0 at the start of a program.
 *
 * The unpacker places sections through the UNP_DEST tile index counter, not through a DEST_TARGET_REG_CFG_MATH_SEC
 * register: those bases are consulted for math-issued instructions only, a write from the unpack thread does not move
 * UNPACR_DEST (emulator, 2026-10-05). The bank therefore lives in this thread's dest_register_offset, and the placers
 * add it, converted to tiles, to every DEST tile index. This zeroes it; @ref _llk_unpack_dest_section_advance_ toggles
 * it per section in SyncHalf.
 *
 * @note Call once per program before the first @ref _llk_unpack_unary_operand_to_dest_init_, never from a per-op init: op writers may
 *       re-run the per-op init inside their tile loop, and a reset there would pin unpack to bank 0 while pack keeps
 *       alternating banks. Pair with @ref _llk_math_pack_sync_init_ (T1) and @ref _llk_pack_dest_init_ (T2).
 */
inline void _llk_unpack_dest_init_()
{
    _reset_dest_register_offset_();
}

/**
 * @brief Move the unpack thread to the other DEST bank after a SyncHalf section.
 *
 * Only this thread's dest_register_offset changes; the placers fold it into the next DEST tile index. No config write
 * and no drain: instructions already issued carry the old index, the toggle only affects indices computed from here on.
 *
 * @tparam EN_32BIT_DEST: Sizes the flip, bank-1 base at row 256 when true (32-bit DEST), 512 when false. Must match
 *         what the pack thread uses, or the two sides address different halves.
 */
template <bool EN_32BIT_DEST>
inline void _llk_unpack_dest_section_advance_()
{
    _update_dest_register_offset_<EN_32BIT_DEST>();
}

/**
 * @brief The current DEST bank base of this thread, in 32x32-tile units of the UNP_DEST tile index counter.
 *
 * dest_register_offset is kept in DEST rows (0, 256 or 512); a 32x32 tile spans 1 << get_dest_tile_size_log2 rows, the
 * same stride @ref _set_dst_write_addr_ uses on the math thread, so the two sides agree on where bank 1 starts.
 */
inline std::uint32_t _llk_unpack_dest_bank_tile_offset_()
{
    return _get_dest_buffer_base_() >> get_dest_tile_size_log2(DstTileShape::Tile32x32);
}

/**
 * @brief MOP configuration for unpacking one tile of a single operand directly into the math DEST register (UNP_DEST).
 *
 * One UNPACR_DEST per run. Both the L1 and DEST tile counters auto-increment (Src/Dst_Tile_Idx_Inc = 1), so back-to-back
 * runs land consecutive tiles at consecutive DEST positions. No dvalid is set: the UNPACK_MATH / MATH_PACK section
 * handshake (@ref _llk_unpack_wait_for_dest_available_ / @ref _llk_unpack_dest_section_done_) takes its place.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer descriptor table;
 *        allocated from the unpack TRISC partition [0,16) at op-init time (see llk_bfd_alloc.h)
 */
inline void _llk_unpack_unary_operand_to_dest_mop_config_(const std::uint32_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    const std::uint32_t unpack_tile_instrn = TT_OP_UNPACR_DEST_TILE_INC(1 /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker to unpack a single operand directly into the math DEST register, synchronized with
 *        math and pack through the UNPACK_PACK / UNPACK_MATH / MATH_PACK semaphores.
 *
 * Unpack-to-dest counterpart of @ref _llk_unpack_unary_operand_init_ (llk_unpack_unary_operand.h). The two families are
 * independent: this one owns its MOP (@ref _llk_unpack_unary_operand_to_dest_mop_config_) and its DEST handshake is the
 * semaphore protocol of the section calls rather than dest-dvalid. Callers pick one family up front.
 *
 * Per-op init: programs the transpose config and the MOP only. It does not touch the DEST bank tracking, so it is
 * safe to call inside a tile loop; the once-per-program bank reset lives in @ref _llk_unpack_dest_init_.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer descriptor table;
 *        allocated from the unpack TRISC partition [0,16) at op-init time (see llk_bfd_alloc.h)
 * @note Transpose is forced off for both unpacker engines: UNP_DEST does not support it. Tiny tiles are not supported.
 * @note Math thread (T1) contract: math runs no datacopy MOP here (skip @ref _llk_math_eltwise_unary_datacopy_init_); it
 *       only forwards UNPACK_MATH into MATH_PACK. Seed all three semaphores on T1 before the first call:
 *       @ref _llk_math_pack_sync_init_ covers MATH_PACK only, so also call @ref _llk_sync_init_ for semaphore::UNPACK_MATH
 *       and semaphore::UNPACK_PACK, each with max N (1 for SyncFull, 2 for SyncHalf) and value 0. Per section, T1 waits on
 *       UNPACK_MATH (@ref _llk_sync_wait_), does its work, posts MATH_PACK (@ref _llk_sync_post_) and gets UNPACK_MATH
 *       (@ref _llk_sync_get_). SyncHalf: T1 flips its own section base (@ref _llk_sync_advance_dest_section_) so its
 *       SFPU work follows the bank the unpacker wrote.
 * @note Pack thread (T2) contract: per section, behind the packer drain, get UNPACK_PACK (frees the bank for this thread)
 *       and then MATH_PACK; SyncHalf: toggle the bank and reprogram the packer's source address offset
 *       (@ref _set_packer_dest_registers_), as the regular path does. DEST_TARGET_REG_CFG_MATH_SEC2 does not move PACR.
 * @note @ref _llk_unpack_dest_init_ must have run once on this thread before the first call. Per section on this
 *       thread: @ref _llk_sync_wait_ STALL_ON_MAX on UNPACK_PACK (a DEST bank is free), any number of
 *       @ref _llk_unpack_unary_operand_to_dest_tile_ / @ref _llk_unpack_unary_operand_to_dest_block_, then, behind an
 *       UNPACK0 drain, @ref _llk_sync_post_ UNPACK_PACK and UNPACK_MATH in that order; SyncHalf:
 *       @ref _llk_unpack_dest_section_advance_.
 *       UNPACK_PACK alone is the unpacker's gate: UNPACK_MATH and MATH_PACK each below N would still admit a third
 *       section into a two-bank DEST (tt-metal #58903).
 */
inline void _llk_unpack_unary_operand_to_dest_init_(const std::uint32_t buf_desc_id)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0 /*TRANSPOSE_EN forced false for UNP_DEST*/);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_unary_operand_to_dest_mop_config_(buf_desc_id);
}

/**
 * @brief Unpacks one tile of a single operand directly into the math DEST register at dst_tile_idx of the current
 *        section. No synchronization.
 *
 * Any number of these calls, each at its own dst_tile_idx, form one section when bracketed by
 * @ref _llk_unpack_wait_for_dest_available_ and @ref _llk_unpack_dest_section_done_.
 *
 * @param l1_tile_idx: Index into the L1 buffer of the tile
 * @param dst_tile_idx: DEST tile index, relative to the current section base. Must be below the section's tile capacity
 *        for the sync mode and DEST width, see @ref get_dest_max_tiles. The pack thread reads the tile back from the same index.
 * @note Call @ref _llk_unpack_unary_operand_to_dest_init_ before this function.
 */
inline void _llk_unpack_unary_operand_to_dest_tile_(const std::uint32_t l1_tile_idx, const std::uint32_t dst_tile_idx)
{
    // UNP_DEST is driven off the UNP_A bank's counters. The DEST bank is part of the tile index (see
    // _llk_unpack_dest_bank_tile_offset_), not of a section base register.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_dest_bank_tile_offset_() + dst_tile_idx);
    ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Unpacks num_tiles consecutive tiles of a single operand directly into DEST tiles
 *        [dst_tile_idx, dst_tile_idx + num_tiles) of the current section. No synchronization.
 *
 * Same as num_tiles calls of @ref _llk_unpack_unary_operand_to_dest_tile_ at consecutive indices, with the counters
 * set once: both auto-increment per MOP run. Bracket with @ref _llk_unpack_wait_for_dest_available_ and
 * @ref _llk_unpack_dest_section_done_ like the tile call.
 *
 * @param l1_tile_idx: Index into the L1 buffer of the first tile of the block
 * @param dst_tile_idx: DEST tile index of the first tile, relative to the current section base
 * @param num_tiles: Tiles in the block. dst_tile_idx + num_tiles must not exceed the section's tile capacity for the
 *        sync mode and DEST width, see @ref get_dest_max_tiles.
 * @note Call @ref _llk_unpack_unary_operand_to_dest_init_ before this function.
 */
inline void _llk_unpack_unary_operand_to_dest_block_(const std::uint32_t l1_tile_idx, const std::uint32_t dst_tile_idx, const std::uint32_t num_tiles)
{
    // UNP_DEST is driven off the UNP_A bank's counters; both auto-increment per tile. The DEST bank is part of the
    // tile index (see _llk_unpack_dest_bank_tile_offset_), not of a section base register.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_dest_bank_tile_offset_() + dst_tile_idx);
    for (std::uint32_t i = 0; i < num_tiles; i++)
    {
        ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}
