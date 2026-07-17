// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_pack_common.h"
#include "tensor_shape.h"

using namespace ckernel;

/**
 * @brief Builds the MOP for packing contiguous tiles from the math destination register via Packer 0.
 *
 * Packer 1 (SrcS / PACR1) is not supported here: it requires autoloop programming; use
 * _llk_pack_srcs_config_ / _llk_pack_srcs_ in llk_srcs.h instead.
 *
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param num_tiles: Number of tiles to pack at a time.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_pack_mop_config_(const std::uint8_t buf_desc_id, const std::uint32_t num_tiles, const TensorShape& tensor_shape)
{
    const std::uint32_t MOP_OUTER_LOOP = num_tiles;
    const std::uint32_t MOP_INNER_LOOP =
        (static_cast<std::uint32_t>(tensor_shape.total_num_faces()) == NUM_FACES) ? 1 : static_cast<std::uint32_t>(tensor_shape.total_num_faces());

    // RT: Use defines to remove these constexpr, and replace with a single TT_OP_PACR_FACE_INC
    std::uint32_t pack_instrn;
    pack_instrn = TT_OP_PACR0_TILE_INC(1 /*Dst_Tile_Idx_Inc*/, 0 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*ClrDatValid*/);

    std::uint32_t incr_to_next_face;
    if (tensor_shape.total_num_faces() < NUM_FACES && tensor_shape.face_r_dim < (FACE_R_DIM >> 1)) // Using sparse tiling: jump to the next index w/ tile
    {
        incr_to_next_face = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, (FACE_R_DIM >> (rows_log2(tensor_shape.face_r_dim) + 1)));
    }
    else // Using dense tiling: just increment to the next tile
    {
        incr_to_next_face = TT_OP_INC_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 1 /*Value*/);
    }

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_instrn, incr_to_next_face);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the packer for packing contiguous tiles via Packer 0.
 *
 * Programs the pack MOP and optionally the packer ReLU (mode and threshold) for Packer 0 via cfg_rmw.
 *
 * @tparam EN_32B_DEST: Set to true when pack reads from the dst register in Float32; controls the
 *         RELU_THRESHOLD register format (32-bit or 16-bit path), values = <true/false>
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 16 - 31
 * @param num_tiles: Number of tiles to pack at a time.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @param relu_config: ReLU config (mode + threshold).
 * @note @ref _llk_pack_ is the matching execute call on this thread.
 */
template <bool EN_32B_DEST = false>
inline void _llk_pack_init_(
    const std::uint8_t buf_desc_id,
    const TensorShape& tensor_shape,
    const std::uint32_t num_tiles          = NUM_TILES,
    const ckernel::ReluConfig& relu_config = ckernel::ReluConfig::none())
{
    _llk_pack_mop_config_(buf_desc_id, num_tiles, tensor_shape);
    _llk_pack_relu_config_<p_pacr::PACK0, EN_32B_DEST>(relu_config);
}

/**
 * @brief Packs out tiles from the math destination register to L1 via Packer 0.
 *
 * @param start_math_dest_tile_idx: The tile index into the math destination register that the packer starts packing from.
 * @param start_l1_tile_idx: The tile index into the L1 output buffer that the packer starts packing into.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @note Call @ref _llk_pack_init_ with matching template args before this function.
 */
inline void _llk_pack_(const std::uint32_t start_math_dest_tile_idx, const std::uint32_t start_l1_tile_idx, const TensorShape& tensor_shape)
{
    //(TODO) RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users offset idx

    // Set Source (math destination) counter to face index offset
    // Set Dest (l1 output) counter to face index offset
    std::uint32_t math_dest_tile_idx = start_math_dest_tile_idx;
    std::uint32_t l1_tile_idx        = start_l1_tile_idx;

    if (tensor_shape.total_num_faces() != NUM_FACES) // using tiny-tiles
    {
        // For face_r_dim >= 8, dest is dense with tiles. For face_r_dim < 8, dest is sparse and tiles are placed every 8 rows.
        // HW defined tiny-tile is registered with 1 face. To map to SW defined tile with different faces, the indices must be multiplied to get the correct
        // offset.
        if (tensor_shape.face_r_dim < (FACE_R_DIM >> 1))
        {
            math_dest_tile_idx *= tensor_shape.total_num_faces() * (FACE_R_DIM >> (rows_log2(tensor_shape.face_r_dim) + 1));
        }
        else
        {
            math_dest_tile_idx *= tensor_shape.total_num_faces();
        }
        l1_tile_idx *= tensor_shape.total_num_faces();
    }

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, math_dest_tile_idx);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, l1_tile_idx);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
