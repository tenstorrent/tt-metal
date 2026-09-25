// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_sync.h"
#include "llk_unpack_unary_operand.h"

using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief Reset the unpack thread's DEST bank tracking to bank 0 at the start of a program.
 *
 * Unpack owns the DEST section base in the unpack-to-dest path: it is the DEST producer (UNP_DEST), so it programs the
 * per-TRISC section base itself rather than letting the math middleman set it on its behalf. TriscID::Unpack selects the
 * same SEC slot the UNP_DEST client reads, regardless of which TRISC this code is compiled for. This establishes the
 * bank-0 base; @ref _llk_unpack_unary_operand_to_dest_ flips it per section (block_ct_dim tiles) in SyncHalf.
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
 * @brief Initializes the unpacker to unpack a single operand directly into the math DEST register, synchronized with
 *        math and pack through the UNPACK_MATH / MATH_PACK semaphores.
 *
 * Unpack-to-dest counterpart of @ref _llk_unpack_unary_operand_init_: it programs the same UNP_DEST MOP
 * (@ref _llk_unpack_unary_operand_mop_config_ with UNP_SEL = UNP_DEST), but the DEST handshake is the semaphore protocol of
 * @ref _llk_unpack_unary_operand_to_dest_ rather than dest-dvalid. Callers pick one family up front; neither branches into the other.
 *
 * Per-op init: programs the transpose config and the MOP only. It does not touch the DEST bank tracking, so it is
 * safe to call inside a tile loop, which op writers do with copy-style inits; the once-per-program bank reset lives in
 * @ref _llk_unpack_dest_init_.
 *
 * Block shape follows the tilize/untilize convention: block_ct_dim is one DEST bank section, the tiles one
 * @ref _llk_unpack_unary_operand_to_dest_ call unpacks before it hands the bank over and (in SyncHalf) moves the section
 * base to the other bank. block_rt_dim, the number of such sections, is the caller's loop around the execute call, as
 * with @ref _llk_unpack_tilize_block_.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer descriptor table;
 *        allocated from the unpack TRISC partition [0,16) at op-init time (see llk_bfd_alloc.h)
 * @param block_ct_dim: Number of tiles per DEST bank section (MOP outer loop length). Must not exceed the DEST bank
 *        capacity for the sync mode and EN_32BIT_DEST stride the execute call uses, see @ref get_dest_max_tiles.
 * @note Transpose is forced off for both unpacker engines: UNP_DEST does not support it. Tiny tiles are not supported.
 * @note Math thread (T1) contract: math runs no datacopy MOP here (skip @ref _llk_math_eltwise_unary_datacopy_init_); it
 *       only forwards UNPACK_MATH into MATH_PACK. Seed both semaphores on T1 before the first call:
 *       @ref _llk_math_pack_sync_init_ covers MATH_PACK only, so also call @ref _llk_sync_init_ (semaphore::UNPACK_MATH,
 *       N, 0) with the same N (1 for SyncFull, 2 for SyncHalf). Per tile, T1 waits on and @ref _llk_sync_get_ UNPACK_MATH,
 *       then posts MATH_PACK with @ref _llk_sync_post_ without flipping the section base (unpack owns it, see
 *       @ref _llk_unpack_dest_init_).
 * @note @ref _llk_unpack_dest_init_ must have run once on this thread before the first call. @ref _llk_unpack_unary_operand_to_dest_ is
 *       the matching execute call on this thread.
 */
inline void _llk_unpack_unary_operand_to_dest_init_(const std::uint32_t buf_desc_id, const std::uint32_t block_ct_dim)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0 /*TRANSPOSE_EN forced false for UNP_DEST*/);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    // IS_32b_DEST_EN only adds the SrcB-clearing NOP for UNP_A/UNP_B; it is ignored for UNP_DEST.
    _llk_unpack_unary_operand_mop_config_<p_unpacr::UNP_DEST, false /*IS_32b_DEST_EN*/>(buf_desc_id, block_ct_dim);
}

/**
 * @brief Unpacks one DEST bank section (block_ct_dim tiles) of a single operand directly into the math DEST register,
 *        synchronized with math and pack through the UNPACK_MATH / MATH_PACK semaphores.
 *
 * One call is one section: wait for a free bank, unpack the block_ct_dim tiles programmed at init starting at DEST tile
 * 0 of the current bank, post UNPACK_MATH, then (SyncHalf) move the section base to the other bank. Callers loop this
 * block_rt_dim times, advancing l1_tile_idx by block_ct_dim per section.
 *
 * The math thread is the middleman with two single-counting semaphores (max = N each). Without an extra wait on
 * MATH_PACK, unpack could race 2N iterations ahead of pack and overwrite a bank that pack has not read yet; waiting on
 * both keeps unpack within N iterations of pack. UNPACK0 is drained before posting UNPACK_MATH so the post does not
 * race the DEST writes math (or pack) will read.
 *
 * @tparam DEST_SYNC_MODE: In SyncHalf, flips the DEST section base to the other bank after each section, values = <SyncFull/SyncHalf>
 * @tparam EN_32BIT_DEST: Sizes the SyncHalf bank flip: bank-1 base at 256 rows when true, 512 when false (see
 *         @ref _update_dest_register_offset_). This is a producer/consumer stride agreement, not a statement about the DEST
 *         mode: it must equal the value the pack thread passes to @ref _llk_sync_advance_dest_section_ for this op, or the
 *         two sides address different DEST halves (the pins live in different TRISC TUs, so no static_assert can compare
 *         them). true is valid in either DEST mode (a 16-bit DEST just uses half of each bank), which is why callers (e.g.
 *         tt-metal's llk_unpack_A) pin it to true to match the pack side. values = <true/false>
 * @param l1_tile_idx: Index into the L1 buffer of the first tile of this section
 * @note Call @ref _llk_unpack_unary_operand_to_dest_init_ before this function. Unpack-to-dest counterpart of
 *       @ref _llk_unpack_unary_operand_.
 */
template <DstSync DEST_SYNC_MODE, bool EN_32BIT_DEST>
inline void _llk_unpack_unary_operand_to_dest_(const std::uint32_t l1_tile_idx)
{
    _llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::MATH_PACK, semaphore::UNPACK_MATH);

    // UNP_DEST is driven off the UNP_A bank's counters.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

    ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);

    // Unpack owns the DEST section base, so it flips to the other bank for the next section.
    if constexpr (DEST_SYNC_MODE == DstSync::SyncHalf)
    {
        _llk_sync_advance_dest_section_<to_underlying(TriscID::Unpack), EN_32BIT_DEST, p_stall::UNPACK0>();
    }
}
