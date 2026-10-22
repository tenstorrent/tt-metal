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
 * @brief Reset the unpack thread's DEST bank tracking to bank 0 at the start of a program.
 *
 * Unpack owns the DEST section base in the unpack-to-dest path: it is the DEST producer (UNP_DEST), so it programs the
 * per-TRISC section base itself rather than letting the math middleman set it on its behalf. TriscID::Unpack selects the
 * same SEC slot the UNP_DEST client reads, regardless of which TRISC this code is compiled for. This establishes the
 * bank-0 base; the section handshake's @ref _llk_sync_advance_dest_section_ flips it per section in SyncHalf
 * (per-section sequence at @ref _llk_unpack_unary_operand_to_dest_init_).
 *
 * @note Call once per program before the first @ref _llk_unpack_unary_operand_to_dest_init_, never from a per-op init: op writers may
 *       re-run the per-op init inside their tile loop, and a reset there would pin unpack to bank 0 while pack keeps
 *       alternating banks. Pair with @ref _llk_math_pack_sync_init_ (T1) and @ref _llk_pack_dest_init_ (T2). The one
 *       exception is the unpack_to_dest compatibility arm of @ref _llk_unpack_unary_operand_init_, which keeps the per-init
 *       reset its unported callers rely on.
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
 * handshake takes its place (per-section sequence at @ref _llk_unpack_unary_operand_to_dest_init_).
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
 * semaphore protocol of the section calls rather than dest-dvalid. The unpack_to_dest = true arm of the unary operand
 * family is a compatibility path for callers not yet ported (the tt-llk test infra and fuser): it keeps that family's
 * init-sized MOP and brackets one @ref _llk_unpack_unary_operand_to_dest_tile_ run with the section handshake itself;
 * new callers use this family directly.
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
 *       thread: @ref _llk_sync_wait_ STALL_ON_MAX on MATH_PACK and UNPACK_MATH (a DEST bank is free), any number of
 *       @ref _llk_unpack_unary_operand_to_dest_tile_ / @ref _llk_unpack_unary_operand_to_dest_block_, then, behind an
 *       UNPACK0 drain, @ref _llk_sync_post_ UNPACK_MATH; SyncHalf: flip the section base
 *       (@ref _llk_sync_advance_dest_section_). The unpack_to_dest arm of @ref _llk_unpack_unary_operand_ runs this
 *       sequence inline; tt-metal runs it in tile_regs_acquire / tile_regs_commit (llk_unpack_wait_for_dest_available /
 *       llk_unpack_dest_section_done, quasar llk_unpack_common_api.h).
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
 * Any number of these calls, each at its own dst_tile_idx, form one section when bracketed by the section handshake
 * (per-section sequence at @ref _llk_unpack_unary_operand_to_dest_init_).
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
 * @brief Unpacks num_tiles tiles of a single operand, l1_tile_stride L1 tiles apart, directly into DEST tiles
 *        [dst_tile_idx, dst_tile_idx + num_tiles) of the current section. No synchronization.
 *
 * Same as num_tiles calls of @ref _llk_unpack_unary_operand_to_dest_tile_ at L1 indices l1_tile_stride apart and
 * consecutive DEST indices: the DEST counter is set once and auto-increments per MOP run, the L1 counter is set per
 * tile. Bracket with the section handshake like the tile call.
 *
 * @param l1_tile_idx: Index into the L1 buffer of the first tile of the block
 * @param dst_tile_idx: DEST tile index of the first tile, relative to the current section base. The pack thread reads
 *        the tiles back from the same indices.
 * @param num_tiles: Tiles in the block. dst_tile_idx + num_tiles must not exceed the section's tile capacity for the
 *        sync mode and DEST width, see @ref get_dest_max_tiles.
 * @param l1_tile_stride: L1 tiles between consecutive tiles of the block: 1 for a contiguous block, the DFB stride when
 *        the buffer interleaves the entries of several threads (tt-metal #56194).
 * @note Call @ref _llk_unpack_unary_operand_to_dest_init_ before this function.
 */
inline void _llk_unpack_unary_operand_to_dest_block_(
    const std::uint32_t l1_tile_idx, const std::uint32_t dst_tile_idx, const std::uint32_t num_tiles, const std::uint32_t l1_tile_stride = 1)
{
    // UNP_DEST is driven off the UNP_A bank's counters. The DEST counter auto-increments per tile; the L1 counter is set
    // per tile, since consecutive tiles of a strided buffer are l1_tile_stride entries apart.
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, dst_tile_idx);
    for (std::uint32_t i = 0; i < num_tiles; i++)
    {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx + i * l1_tile_stride);
        ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}
