// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"

namespace ckernel
{

// Fast tilize processes tiles in pairs (unit_dim 2) except for a width-1 block (unit_dim 1) and an odd tail
// (unit_dim 3). Only the pair unit has a two-face (16x32 tiny tile) data path.
constexpr std::uint32_t FAST_TILIZE_PAIR_UNIT_DIM       = 2;
constexpr std::uint32_t FAST_TILIZE_TINY_TILE_NUM_FACES = 2;

constexpr const char* FAST_TILIZE_TINY_TILE_UNIT_DIM_MSG = "16x32 tiny tiles are only supported for tensors with even-sized tile widths for fast_tilize";

/**
 * @brief Whether a fast-tilize unit of unit_dim tiles can process tiles with num_faces faces.
 *
 * Four-face (32x32) tiles work with every unit_dim; two-face (16x32) tiles only with the pair unit.
 *
 * @param unit_dim: Tiles processed per iteration, valid values = <1, 2, 3>.
 * @param num_faces: Faces per tile, valid values = <2, 4>.
 */
constexpr bool fast_tilize_unit_supports_faces(const std::uint32_t unit_dim, const std::uint32_t num_faces)
{
    return num_faces == TILE_NUM_FACES || (num_faces == FAST_TILIZE_TINY_TILE_NUM_FACES && unit_dim == FAST_TILIZE_PAIR_UNIT_DIM);
}

/**
 * @brief Whether the Wormhole fast-tilize path supports a block of tiles of the given geometry and width.
 *
 * Faces must be 16 rows high: the unpack, math and pack sequences are written for 16x16 faces and never
 * read the face height. 32x32 tiles (four faces) work at any width. 16x32 tiny tiles (two faces) work
 * only when the whole block is processed in pairs, i.e. at an even width (a width of 1 or an odd tail
 * would need unit_dim 1 or 3, which have no two-face path). Any other geometry must take the plain tilize
 * path: 32x16 (also two faces but 16 wide), or a 16x32 tile declared as four 8x16 faces, which the
 * num_faces == 4 sequences would treat as 32 rows high and read and write past its pages.
 *
 * @param tile_r_dim: Tile height in datums of the operand being tilized.
 * @param tile_c_dim: Tile width in datums of the operand being tilized.
 * @param face_r_dim: Face height in datums of the operand being tilized.
 * @param num_faces: Faces per tile of the operand being tilized.
 * @param block: Block width in tiles.
 */
constexpr bool fast_tilize_block_supported(
    const std::uint32_t tile_r_dim, const std::uint32_t tile_c_dim, const std::uint32_t face_r_dim, const std::uint32_t num_faces, const std::uint32_t block)
{
    return face_r_dim == FACE_R_DIM && tile_c_dim == TILE_C_DIM &&
           ((tile_r_dim == TILE_R_DIM && num_faces == TILE_NUM_FACES) ||
            (tile_r_dim == FACE_R_DIM && num_faces == FAST_TILIZE_TINY_TILE_NUM_FACES && block % FAST_TILIZE_PAIR_UNIT_DIM == 0));
}

} // namespace ckernel
