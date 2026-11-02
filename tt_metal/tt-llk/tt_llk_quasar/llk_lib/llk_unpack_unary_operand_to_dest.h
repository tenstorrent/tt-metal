// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @brief DEST rows one tile of tensor_shape spans: the per-tile stride math and pack use for the shape.
 *
 * A 4-face tile spans 64 rows. A tiny tile is laid out one face per 8-row slot at least, so 1x32..8x32 span 16 rows,
 * 16x32 32, as DstTileShape sizes them (get_dest_tile_size_log2); 16x16 spans 16 and 32x16 32.
 *
 * @param tensor_shape: Tile shape
 */
constexpr std::uint32_t _llk_unpack_to_dest_tile_rows_(const TensorShape& tensor_shape)
{
    return tensor_shape.total_num_faces() * tensor_shape.face_r_dim * tiny_face_stride(tensor_shape);
}

/**
 * @brief UNP_DEST tile counter value for DEST tile dst_tile_idx of the current bank.
 *
 * In units of the buffer descriptor's tile: a 4-face tile is one descriptor tile of 64 rows, a tiny tile is described
 * one face per descriptor tile of face_r_dim rows, so the index is scaled the way the pack side scales its source index
 * (@ref _llk_pack_). The DEST bank is part of the index: the SEC base registers do not move UNPACR_DEST.
 *
 * @param tensor_shape: Tile shape
 * @param dst_tile_idx: DEST tile index, relative to the current section base
 */
inline std::uint32_t _llk_unpack_to_dest_dst_tile_counter_(const TensorShape& tensor_shape, const std::uint32_t dst_tile_idx)
{
    if (tensor_shape.total_num_faces() == NUM_FACES)
    {
        return _llk_unpack_dest_bank_tile_offset_() + dst_tile_idx;
    }
    const std::uint32_t tiles_per_tile = tensor_shape.total_num_faces() * tiny_face_stride(tensor_shape);
    return (_get_dest_buffer_base_() >> rows_log2(tensor_shape.face_r_dim)) + dst_tile_idx * tiles_per_tile;
}

/**
 * @brief UNP_A source tile counter value for L1 tile l1_tile_idx: a tiny tile is one descriptor tile per face.
 *
 * @param tensor_shape: Tile shape
 * @param l1_tile_idx: Index into the L1 buffer of the tile
 */
constexpr std::uint32_t _llk_unpack_to_dest_src_tile_counter_(const TensorShape& tensor_shape, const std::uint32_t l1_tile_idx)
{
    return tensor_shape.total_num_faces() == NUM_FACES ? l1_tile_idx : l1_tile_idx * tensor_shape.total_num_faces();
}

/**
 * @brief MOP configuration for unpacking one tile of a single operand directly into the math DEST register (UNP_DEST).
 *
 * One UNPACR_DEST per descriptor tile: one per run for a 4-face tile, one per face for a tiny tile. Both the L1 and
 * DEST tile counters auto-increment, the DEST one by tiny_face_stride so faces shorter than 8 rows take 8-row slots,
 * the layout the pack side reads (@ref _llk_pack_mop_config_); back-to-back runs land consecutive tiles at consecutive
 * DEST positions. No dvalid is set: the UNPACK_MATH / MATH_PACK section handshake takes its place (per-section sequence
 * at @ref _llk_unpack_unary_operand_to_dest_init_).
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer descriptor table;
 *        allocated from the unpack TRISC partition [0,16) at op-init time (see llk_bfd_alloc.h)
 * @param tensor_shape: Tile shape the descriptor was programmed with
 */
inline void _llk_unpack_unary_operand_to_dest_mop_config_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = tensor_shape.total_num_faces() == NUM_FACES ? 1 : tensor_shape.total_num_faces();

    const std::uint32_t unpack_tile_instrn =
        TT_OP_UNPACR_DEST_TILE_INC(tiny_face_stride(tensor_shape) /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker to unpack a single operand directly into the math DEST register, synchronized with
 *        math and pack through the UNPACK_PACK / UNPACK_MATH / MATH_PACK semaphores.
 *
 * Unpack-to-dest counterpart of @ref _llk_unpack_unary_operand_init_ (llk_unpack_unary_operand.h), which only unpacks to
 * SrcA / SrcB. The two families are independent: this one owns its MOP (@ref _llk_unpack_unary_operand_to_dest_mop_config_)
 * and its DEST handshake is the semaphore protocol of the section calls, or dest-dvalid (see the last @note).
 *
 * Per-op init: programs the transpose config and the MOP only. It does not touch the DEST bank tracking, so it is
 * safe to call inside a tile loop; the once-per-program bank reset lives in @ref _llk_unpack_dest_init_.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer descriptor table;
 *        allocated from the unpack TRISC partition [0,16) at op-init time (see llk_bfd_alloc.h)
 * @param tensor_shape: Tile shape the descriptor was programmed with (Continuous L1 access); defaults to 32x32. Tiny
 *        tiles land at the DEST layout the pack side reads (@ref _llk_unpack_to_dest_tile_rows_); pass the same shape
 *        to the tile / block calls.
 * @note Transpose is forced off for both unpacker engines: UNP_DEST does not support it.
 * @note Math thread (T1) contract: math runs no datacopy MOP here (skip @ref _llk_math_eltwise_unary_datacopy_init_); it
 *       only forwards UNPACK_MATH into MATH_PACK. Once per program, @ref _llk_math_pack_sync_init_ (seeds MATH_PACK, the
 *       semaphore T1 posts). Right before T1's first UNPACK_MATH wait, not at init, @ref _llk_sync_spin_until_posted_
 *       (semaphore::UNPACK_MATH), so the wait is armed only after T0's seed has landed; a program with no sections
 *       never posts, so a spin at init would never end. Per section, T1 waits on UNPACK_MATH
 *       (@ref _llk_sync_wait_), does its work, posts MATH_PACK (@ref _llk_sync_post_) and gets UNPACK_MATH
 *       (@ref _llk_sync_get_). SyncHalf: T1 flips its own section base (@ref _llk_sync_advance_dest_section_) so its
 *       SFPU work follows the bank the unpacker wrote.
 * @note Pack thread (T2) contract: per section, behind the packer drain, get UNPACK_PACK (frees the bank for the unpack
 *       thread, T0) and then MATH_PACK; SyncHalf: toggle the bank and reprogram the packer's source address offset
 *       (@ref _set_packer_dest_registers_), as the regular path does. DEST_TARGET_REG_CFG_MATH_SEC2 does not move PACR.
 * @note @ref _llk_unpack_dest_init_ must have run once on this thread before the first call, followed by the seeds of
 *       the semaphores this thread posts, @ref _llk_sync_init_ for semaphore::UNPACK_MATH and semaphore::UNPACK_PACK,
 *       each with max N (1 for SyncFull, 2 for SyncHalf) and value 0. Per section on this
 *       thread: @ref _llk_sync_wait_ STALL_ON_MAX on UNPACK_PACK (a DEST bank is free), any number of
 *       @ref _llk_unpack_unary_operand_to_dest_tile_ / @ref _llk_unpack_unary_operand_to_dest_block_, then, behind an
 *       UNPACK0 drain, @ref _llk_sync_post_ UNPACK_PACK and UNPACK_MATH in that order; SyncHalf:
 *       @ref _llk_unpack_dest_section_advance_. UNPACK_PACK alone is the unpacker's gate: UNPACK_MATH and MATH_PACK
 *       each below N would still admit a third section into a two-bank DEST (tt-metal #58903). tt-metal runs this
 *       sequence in tile_regs_acquire / tile_regs_commit (llk_unpack_wait_for_dest_available /
 *       llk_unpack_dest_section_done, quasar llk_unpack_common_api.h).
 * @note Dest-dvalid form (the standalone tt-llk tests): the tile / block calls carry no synchronization, so they can be
 *       bracketed with the dest-dvalid handshake instead. Run @ref _llk_unpack_dest_init_ once so the bank tile offset
 *       stays 0, never call @ref _llk_unpack_dest_section_advance_ (dest-dvalid moves this client between banks in
 *       hardware), and close each section with @ref _llk_unpack_dest_dvalid_section_done_.
 */
inline void _llk_unpack_unary_operand_to_dest_init_(const std::uint32_t buf_desc_id, const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0 /*TRANSPOSE_EN forced false for UNP_DEST*/);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_unary_operand_to_dest_mop_config_(buf_desc_id, tensor_shape);
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
 *        for the sync mode, DEST width and tile shape (@ref get_dest_max_tiles, @ref _llk_unpack_to_dest_tile_rows_).
 *        The pack thread reads the tile back from the same index.
 * @param tensor_shape: Tile shape passed to @ref _llk_unpack_unary_operand_to_dest_init_; defaults to 32x32
 * @note Call @ref _llk_unpack_unary_operand_to_dest_init_ before this function.
 */
inline void _llk_unpack_unary_operand_to_dest_tile_(
    const std::uint32_t l1_tile_idx, const std::uint32_t dst_tile_idx, const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    // UNP_DEST is driven off the UNP_A bank's counters, in descriptor tiles (see the counter helpers above).
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_to_dest_src_tile_counter_(tensor_shape, l1_tile_idx));
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_to_dest_dst_tile_counter_(tensor_shape, dst_tile_idx));
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
 *        sync mode, DEST width and tile shape (@ref get_dest_max_tiles, @ref _llk_unpack_to_dest_tile_rows_).
 * @param l1_tile_stride: L1 tiles between consecutive tiles of the block: 1 for a contiguous block, the DFB stride when
 *        the buffer interleaves the entries of several threads (tt-metal #56194).
 * @param tensor_shape: Tile shape passed to @ref _llk_unpack_unary_operand_to_dest_init_; defaults to 32x32
 * @note Call @ref _llk_unpack_unary_operand_to_dest_init_ before this function.
 */
inline void _llk_unpack_unary_operand_to_dest_block_(
    const std::uint32_t l1_tile_idx,
    const std::uint32_t dst_tile_idx,
    const std::uint32_t num_tiles,
    const std::uint32_t l1_tile_stride = 1,
    const TensorShape& tensor_shape    = DEFAULT_TENSOR_SHAPE)
{
    // UNP_DEST is driven off the UNP_A bank's counters, in descriptor tiles (see the counter helpers above). The DEST
    // counter auto-increments one tile per run; the L1 counter is set per tile, since consecutive tiles of a strided
    // buffer are l1_tile_stride entries apart.
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_to_dest_dst_tile_counter_(tensor_shape, dst_tile_idx));
    for (std::uint32_t i = 0; i < num_tiles; i++)
    {
        TT_SET_SRC_TILE_FACE_ROW_IDX(
            p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_to_dest_src_tile_counter_(tensor_shape, l1_tile_idx + i * l1_tile_stride));
        ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}
