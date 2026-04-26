// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_pack_common.h"

using namespace ckernel;

/**
 * @brief MOP configuration for pack of contiguous tiles
 * @details Sets up MOP for packing out tile by tile from the math destination register via Packer 0.
 * Packer 1 (SrcS / PACR1) is not supported here: it requires autoloop programming; use
 * _llk_pack_srcs_config_ / _llk_pack_srcs_ in llk_srcs_tdma.h instead.
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param num_tiles: number of tiles to pack at a time
 * @param num_faces: number of faces in the tiles to unpack, default to NUM_FACES
 */
template <std::uint8_t PACK_SEL>
inline void _llk_pack_mop_config_(const std::uint8_t buf_desc_id, const std::uint32_t num_tiles, const TileShape& tile_shape)
{
    static_assert((PACK_SEL == p_pacr::PACK0) || (PACK_SEL == p_pacr::PACK1), "PACK_SEL can only be set to p_pacr::PACK0/PACK1");

    const std::uint32_t MOP_OUTER_LOOP = num_tiles;
    const std::uint32_t MOP_INNER_LOOP = (tile_shape.num_faces == NUM_FACES) ? 1 : tile_shape.num_faces;

    // RT: Use defines to remove these constexpr, and replace with a single TT_OP_PACR_FACE_INC
    std::uint32_t pack_instrn;
    if constexpr (PACK_SEL == p_pacr::PACK0)
    {
        pack_instrn = TT_OP_PACR0_TILE_INC(1 /*Dest Tile Idx*/, 0 /*Src Tile Idx*/, buf_desc_id, 0);
    }
    else
    {
        // PACR1 slice size is only 16x8, no need to increment SrcS counter
        pack_instrn = TT_OP_PACR1_TILE_INC(1 /*Dest Tile Idx*/, 0, buf_desc_id, 0);
    }

    std::uint32_t incr_to_next_face;
    if (tile_shape.num_faces < NUM_FACES && tile_shape.face_r_dim < (FACE_R_DIM >> 1)) // Using sparse tiling: jump to the next index w/ tile
    {
        incr_to_next_face = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, PACK_SEL, (FACE_R_DIM >> (rows_log2(tile_shape.face_r_dim) + 1)));
    }
    else // Using dense tiling: just increment to the next tile
    {
        incr_to_next_face = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, PACK_SEL, 1);
    }

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, incr_to_next_face);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialization for pack of contiguous tiles
 * @details Sets up MOP for packing out tile by tile via Packer 0.
 * Optionally programs packer ReLU (MODE and THRESHOLD) for Packer 0 via cfg_rmw.
 * @tparam EN_32B_DEST: Set to true when pack reads from dst register in Float32;
 * controls RELU_THRESHOLD register format (32-bit or 16-bit path).
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 16-31
 * @param num_tiles: number of tiles to pack at a time
 * @param num_faces: number of faces in the tiles to pack, default to NUM_FACES
 * @param relu_config ReLU config (mode + threshold).
 */
template <std::uint8_t PACK_SEL, bool EN_32B_DEST = false>
inline void _llk_pack_init_(
    const std::uint8_t buf_desc_id,
    const TileShape& tile_shape,
    const std::uint32_t num_tiles          = NUM_TILES,
    const ckernel::ReluConfig& relu_config = ckernel::ReluConfig::none())
{
    _llk_pack_mop_config_<PACK_SEL>(buf_desc_id, num_tiles, tile_shape);
    _llk_pack_relu_config_<PACK_SEL, EN_32B_DEST>(relu_config);
}

/**
 * @brief Packs out tiles from the math destination register via Packer 0
 * @param start_math_dest_tile_idx: The tile index into the math destination register
 * that packer can start packing from
 * @param start_l1_tile_idx: The tile index into the l1 output buffer
 * that packer can start packing into
 */
template <std::uint8_t PACK_SEL>
inline void _llk_pack_(const std::uint32_t start_math_dest_tile_idx, const std::uint32_t start_l1_tile_idx, const TileShape& tile_shape)
{
    //(TODO) RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users offset idx

    // Set Source (math destination) counter to face index offset
    // Set dst (l1 output) counter to face index offset
    if (tile_shape.num_faces == NUM_FACES) // Using full tiles
    {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, PACK_SEL, start_math_dest_tile_idx);
        TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, PACK_SEL, start_l1_tile_idx);
    }
    else // Using tiny-tiles
    {
        // For face_r_dim >= 8, dest is dense with tiles. For face_r_dim < 8, dest is sparse and tiles are placed every 8 rows.
        // HW defined tiny-tile is registered with 1 face. To map to SW defined tile with different faces, the indices must be multiplied to get the correct
        // offset.
        if (tile_shape.face_r_dim < (FACE_R_DIM >> 1))
        {
            TT_SET_SRC_TILE_FACE_ROW_IDX(
                p_set_inc_sel::TILE_SEL, PACK_SEL, start_math_dest_tile_idx * tile_shape.num_faces * (FACE_R_DIM >> (rows_log2(tile_shape.face_r_dim) + 1)));
        }
        else
        {
            TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, PACK_SEL, start_math_dest_tile_idx * tile_shape.num_faces);
        }
        TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, PACK_SEL, start_l1_tile_idx * tile_shape.num_faces);
    }
    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
