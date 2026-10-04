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
 * Unpack owns the DEST section base in the unpack-to-dest path: it is the DEST producer (UNP_DEST), so it programs the
 * per-TRISC section base itself rather than letting the math middleman set it on its behalf. TriscID::Unpack selects the
 * same SEC slot the UNP_DEST client reads, regardless of which TRISC this code is compiled for. This establishes the
 * bank-0 base; @ref _llk_unpack_dest_section_done_ flips it per section in SyncHalf.
 *
 * @note Call once per program before the first @ref _llk_unpack_unary_operand_to_dest_init_, never from a per-op init: op writers may
 *       re-run the per-op init inside their tile loop, and a reset there would pin unpack to bank 0 while pack keeps
 *       alternating banks. Pair with @ref _llk_math_pack_sync_init_ (T1) and @ref _llk_pack_dest_init_ (T2).
 */
inline void _llk_unpack_dest_init_()
{
    _reset_dest_register_offset_();
    _set_dest_section_base_<to_underlying(TriscID::Unpack)>(_get_dest_buffer_base_());
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
 *        math and pack through the UNPACK_MATH / MATH_PACK semaphores.
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
 *       only forwards UNPACK_MATH into MATH_PACK. Seed both semaphores on T1 before the first call:
 *       @ref _llk_math_pack_sync_init_ covers MATH_PACK only, so also call @ref _llk_sync_init_ (semaphore::UNPACK_MATH,
 *       N, 0) with the same N (1 for SyncFull, 2 for SyncHalf). Per section, T1 waits on UNPACK_MATH (@ref _llk_sync_wait_),
 *       does its work, posts MATH_PACK (@ref _llk_sync_post_) and only then gets UNPACK_MATH (@ref _llk_sync_get_), so the
 *       section stays counted in one of the two semaphores from unpack write to pack release. T1 never flips the
 *       section base (unpack owns it, see @ref _llk_unpack_dest_init_).
 * @note @ref _llk_unpack_dest_init_ must have run once on this thread before the first call. Per section on this
 *       thread: @ref _llk_unpack_wait_for_dest_available_, any number of @ref _llk_unpack_unary_operand_to_dest_tile_ /
 *       @ref _llk_unpack_unary_operand_to_dest_block_, then @ref _llk_unpack_dest_section_done_.
 */
inline void _llk_unpack_unary_operand_to_dest_init_(const std::uint32_t buf_desc_id)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0 /*TRANSPOSE_EN forced false for UNP_DEST*/);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_unary_operand_to_dest_mop_config_(buf_desc_id);
}

/**
 * @brief Wait until a DEST bank is free for the unpack thread to write: unpack-to-dest section begin.
 *
 * Unpack side of the DEST section handshake, the counterpart of @ref _llk_math_wait_for_dest_available_ and
 * @ref _llk_packer_wait_for_math_done_. Math is the middleman with two semaphores of max N (1 in SyncFull, 2 in
 * SyncHalf): UNPACK_MATH counts sections unpacked but not yet committed by math, MATH_PACK counts sections committed
 * but not yet released by pack. Waiting on both keeps unpack within N sections of pack.
 *
 * @note One call per section, paired with one @ref _llk_unpack_dest_section_done_. The section is the unit on all
 *       three threads: however many tiles the section holds, each thread posts or gets once.
 * @note In SyncHalf the per-semaphore wait still admits UNPACK_MATH = 1 and MATH_PACK = 1 at once, i.e. a third section
 *       into a two-bank DEST (tt-metal #58903). A DEST occupancy count that unpack posts and pack gets closes that.
 */
inline void _llk_unpack_wait_for_dest_available_()
{
    _llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::MATH_PACK, semaphore::UNPACK_MATH);
}

/**
 * @brief Hand the current DEST section to math: unpack-to-dest section end.
 *
 * Posts UNPACK_MATH once UNPACK0 has drained, so the post cannot overtake the DEST writes math and pack will read, then
 * (SyncHalf) moves the unpack thread's section base to the other bank. Counterpart of @ref _llk_math_dest_section_done_
 * and @ref _llk_pack_dest_semaphore_section_done_.
 *
 * @tparam DEST_SYNC_MODE: In SyncHalf, flips the DEST section base to the other bank after the section, values = <SyncFull/SyncHalf>
 * @tparam EN_32BIT_DEST: Sizes the SyncHalf bank flip: bank-1 base at 256 rows when true, 512 when false (see
 *         @ref _update_dest_register_offset_). Must equal the value the pack thread passes to
 *         @ref _llk_sync_advance_dest_section_ for this op, or the two sides address different DEST halves (the two
 *         calls live in different TRISC TUs, so no static_assert can compare them). values = <true/false>
 */
template <DstSync DEST_SYNC_MODE, bool EN_32BIT_DEST>
inline void _llk_unpack_dest_section_done_()
{
    _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);
    if constexpr (DEST_SYNC_MODE == DstSync::SyncHalf)
    {
        _llk_sync_advance_dest_section_<to_underlying(TriscID::Unpack), EN_32BIT_DEST, p_stall::UNPACK0>();
    }
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
    // UNP_DEST is driven off the UNP_A bank's counters.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, dst_tile_idx);
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
    // UNP_DEST is driven off the UNP_A bank's counters; both auto-increment per tile.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, dst_tile_idx);
    for (std::uint32_t i = 0; i < num_tiles; i++)
    {
        ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}
