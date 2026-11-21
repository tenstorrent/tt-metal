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
 * @brief MOP configuration for unpacking one tile of a single operand directly into the math DEST register (UNP_DEST).
 *
 * Issues one UNPACR_DEST per run. Both the L1 and DEST tile counters advance automatically, so consecutive runs
 * write consecutive tiles to DEST. The caller handles synchronization. This MOP does not set dvalid.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer descriptor table;
 *        the fixed per-engine BFD id (ckernel_buf_desc.h)
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
 * @brief Initializes the unpacker to unpack a single operand directly into a borrowed DEST section.
 *
 * Like @ref _llk_unpack_unary_operand_init_, but writes to DEST instead of SrcA or SrcB. This path has its own MOP,
 * configured by @ref _llk_unpack_unary_operand_to_dest_mop_config_. Use either the borrowing handshake or the Dest
 * dvalid handshake to synchronize access to DEST, as described below.
 *
 * This init programs only the transpose config and the MOP. It leaves the DEST bank tracking unchanged, so it can
 * run inside a tile loop. Call @ref _llk_unpack_dest_init_ once per program to reset the bank tracking.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is stored in the buffer descriptor table;
 *        the fixed per-engine BFD id (ckernel_buf_desc.h)
 * @note Transpose is forced off for both unpacker engines: UNP_DEST does not support it. Tiny tiles are not supported.
 * @note With semaphores, MATH or PACK lends its section through _llk_tile_regs_borrow_<FROM, UNPACK>.
 *       Both threads must call borrow before the tile or block operations and _llk_tile_regs_return_ afterward.
 *       The lender provides the exact base, so unpack does not switch banks on its own. The lender keeps ownership
 *       through MATH_PACK and advances its bank through the normal commit or release call after the loan ends.
 * @note Standalone tests can use the Dest dvalid handshake instead. Call @ref _llk_unpack_dest_init_ once and keep
 *       the software bank offset at zero. Do not call @ref _llk_unpack_dest_section_advance_, since Dest dvalid
 *       switches the bank in hardware. Finish each section with @ref _llk_unpack_dest_dvalid_section_done_.
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
    // UNP_DEST is driven off the UNP_A bank's counters. The DEST bank is part of the tile index (see
    // _llk_unpack_dest_bank_tile_offset_), not of a section base register.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_dest_bank_tile_offset_() + dst_tile_idx);
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
    // per tile, since consecutive tiles of a strided buffer are l1_tile_stride entries apart. The DEST bank is part of
    // the tile index (see _llk_unpack_dest_bank_tile_offset_), not of a section base register.
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, _llk_unpack_dest_bank_tile_offset_() + dst_tile_idx);
    for (std::uint32_t i = 0; i < num_tiles; i++)
    {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx + i * l1_tile_stride);
        ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }
}
