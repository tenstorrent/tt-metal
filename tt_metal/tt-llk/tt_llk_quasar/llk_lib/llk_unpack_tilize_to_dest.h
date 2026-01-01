// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_sync.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::trisc;

/**
 * @file llk_unpack_tilize_to_dest.h
 * @brief Unpack-tilize of full 32x32 tiles straight into the math DEST register (UNP_DEST).
 *
 * Unpack-to-dest counterpart of the SrcA/SrcB family in llk_unpack_tilize.h. The unpacker is the DEST producer: math
 * runs no datacopy and only forwards each section, pack reads it out, and the three threads hand off through the
 * UNPACK_PACK / UNPACK_MATH / MATH_PACK semaphores, the protocol documented on @ref _llk_unpack_unary_operand_to_dest_init_.
 * UNP_DEST is driven through UNPACKER0's config and counters, so the UNP_A registers are programmed throughout.
 */

/**
 * @brief MOP for unpack-tilize into DEST: one UNPACR_TILIZE per tile, num_tiles tiles per run.
 *
 * The source Z counter steps one tile in L1 and the DEST Z counter one tile in DEST (DST_Z_STRIDE, see
 * @ref _llk_unpack_tilize_to_dest_init_), so a run lands its tiles at consecutive DEST tile indices. The last tile's
 * instruction resets the counters instead; every run sets them explicitly first.
 *
 * @param buf_desc_id: Buffer descriptor id, allocated from the unpack TRISC partition at op-init time (llk_bfd_alloc.h)
 * @param num_tiles: Tiles per MOP run
 */
inline void _llk_unpack_tilize_to_dest_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
{
    LLK_ASSERT(num_tiles <= MOP_LOOP_LEN_MAX, "num_tiles exceeds the MOP loop length");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    const std::uint32_t MOP_INNER_LOOP     = num_tiles;

    const std::uint32_t unpack_tile_instrn = TT_OP_UNPACR_TILIZE(0, 0, 1 /*dst Z inc*/, 1 /*src Z inc*/, p_unpacr::UNP_DEST, buf_desc_id, 0 /*no dvalid*/);
    const std::uint32_t reset_cntr_instrn =
        TT_OP_UNPACR_TILIZE(0, 1 /*Cntr_Reset_Mask*/, 0 /*dst Z inc*/, 0 /*src Z inc*/, p_unpacr::UNP_DEST, buf_desc_id, 0 /*no dvalid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_tile_instrn);
    temp.set_last_outer_loop_instr(reset_cntr_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker to unpack-tilize full 32x32 tiles straight into DEST.
 *
 * Programs the UNPACKER0 tilize strides, with DST_Z_STRIDE set to one tile so each DEST Z step is one tile, and the
 * block_ct_dim-tile MOP. Per-op init: it does not touch the DEST bank tracking, so it is safe to call inside a tile loop; the once-per-program
 * bank reset is @ref _llk_unpack_dest_init_.
 *
 * @param buf_desc_id: Buffer descriptor id, allocated from the unpack TRISC partition at op-init time (llk_bfd_alloc.h)
 * @param full_ct_dim: Tiles in a row of the row-major input tensor
 * @param block_ct_dim: Tiles per @ref _llk_unpack_tilize_to_dest_ run, a divisor of full_ct_dim; at most the DEST section
 *        capacity
 * @param tensor_shape: Tile shape; 32x32 tiles only
 * @note Math thread (T1): no datacopy init, it only forwards UNPACK_MATH into MATH_PACK per section. Pack thread (T2):
 *       @ref _llk_pack_init_, per section get UNPACK_PACK then MATH_PACK. On this thread, per section: wait
 *       STALL_ON_MAX on UNPACK_PACK, any number of runs within the section capacity (@ref get_dest_max_tiles), then
 *       behind an UNPACK0 drain post UNPACK_PACK and UNPACK_MATH in that order; SyncHalf:
 *       @ref _llk_unpack_dest_section_advance_.
 */
inline void _llk_unpack_tilize_to_dest_init_(
    const std::uint32_t buf_desc_id, const std::uint32_t full_ct_dim, const std::uint32_t block_ct_dim, const TensorShape& tensor_shape)
{
    LLK_ASSERT(tensor_shape.total_num_faces() == NUM_FACES && tensor_shape.face_r_dim == FACE_R_DIM, "unpack-tilize to DEST supports 32x32 tiles only");
    LLK_ASSERT(block_ct_dim >= 1 && full_ct_dim % block_ct_dim == 0, "block_ct_dim must divide full_ct_dim");

    ckernel::unpack::unpack_tilize_cfg_u unpk_cfg = {};
    unpk_cfg.f.src_z_stride                       = tensor_shape.num_faces_c_dim;                             // one tile in L1, in faces
    unpk_cfg.f.dst_z_stride                       = tensor_shape.total_num_faces() * tensor_shape.face_r_dim; // one tile in DEST, in rows
    unpk_cfg.f.stride_val_source                  = 0;
    unpk_cfg.f.stride_offset_0                    = full_ct_dim * tensor_shape.num_faces_c_dim; // next row within the same tile

    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0); // UNP_DEST does not support transpose
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    cfg[THCON_UNPACKER0_REG1_UNPACK_TILIZE_SRC_Z_STRIDE_ADDR32] = unpk_cfg.val[0];
    cfg[THCON_UNPACKER0_REG2_UNPACK_STRIDE_OFFSET_0_ADDR32]     = unpk_cfg.val[2];

    _llk_unpack_tilize_to_dest_mop_config_(buf_desc_id, block_ct_dim);
}

/**
 * @brief Sets the L1 offset of the current row of tiles: the Z counter walks tiles across a row, not between rows.
 *
 * Drains UNPACK0 first, since the previous row's UNPACR_TILIZE may still be reading this CFG word.
 *
 * @param tensor_shape: Tile shape
 * @param l1_base_idx: Offset of the current row of tiles, in tile-width rows of the L1 buffer
 * @note Call once per row of tiles, before the @ref _llk_unpack_tilize_to_dest_ calls that tilize that row.
 */
inline void _llk_unpack_tilize_to_dest_set_src_offset_(const TensorShape& tensor_shape, const std::uint32_t l1_base_idx)
{
    _llk_stall_cfg_on_<p_stall::UNPACK0>();
    cfg_rmw(THCON_UNPACKER0_REG0_TILIZE_SRC_ADDR_OFFSET_RMW, l1_base_idx * tensor_shape.num_faces_c_dim);
}

/**
 * @brief Points the L1 and DEST counters at the first tile of a run.
 *
 * @param l1_tile_idx: Column index of the first tile within the row set by @ref _llk_unpack_tilize_to_dest_set_src_offset_
 * @param dst_tile_idx: DEST tile index of the first tile, relative to the section base
 */
inline void _llk_unpack_tilize_to_dest_set_counters_(const std::uint32_t l1_tile_idx, const std::uint32_t dst_tile_idx)
{
    // Only 11 bits of the value reach the unpacker under FACE_SEL; anything larger aliases back to an earlier tile.
    LLK_ASSERT(l1_tile_idx < FACE_SEL_IDX_LIMIT, "l1_tile_idx exceeds the 11 bits the src face index can carry");

    // UNPACR_TILIZE addresses DEST through the Z (face) counter, one tile per step, so the DEST bank is folded into that
    // index: the SEC base registers do not move the unpacker (see _llk_unpack_dest_bank_tile_offset_). The tile counter
    // is zeroed in case a preceding op left it auto-incremented.
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::FACE_SEL, p_unpacr::UNP_A, _llk_unpack_dest_bank_tile_offset_() + dst_tile_idx);
}

/**
 * @brief Unpack-tilizes block_ct_dim tiles of the current row into DEST tiles [dst_tile_idx, dst_tile_idx +
 *        block_ct_dim) of the current section. No synchronization: bracket with the section handshake.
 *
 * @param l1_tile_idx: Column index of the first tile within the row set by @ref _llk_unpack_tilize_to_dest_set_src_offset_
 * @param dst_tile_idx: DEST tile index of the first tile, relative to the section base. dst_tile_idx + block_ct_dim must
 *        not exceed the section capacity (@ref get_dest_max_tiles). Pack reads the tiles back from the same indices.
 * @note Call @ref _llk_unpack_tilize_to_dest_init_ before this function.
 */
inline void _llk_unpack_tilize_to_dest_(const std::uint32_t l1_tile_idx, const std::uint32_t dst_tile_idx)
{
    _llk_unpack_tilize_to_dest_set_counters_(l1_tile_idx, dst_tile_idx);
    ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
