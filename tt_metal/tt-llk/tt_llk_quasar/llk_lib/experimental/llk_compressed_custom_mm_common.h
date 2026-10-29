// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cfg_defines.h"
#include "tensix_types.h"
#include "tensor_shape.h"

/*************************************************************************
 * LLK COMPRESSED CUSTOM MM - shared contract
 *************************************************************************/

// Quasar port of the Blackhole compressed_custom_mm: Input0 [M, kt_dim * 32] x Input1 [kt_dim * 32, ct_dim * 32],
// where every Input1 (weight) tile carries its own L1 format and all-zero weight tiles are not stored at all.
//
// Limitations, same as the Blackhole op unless stated:
//   in0 tile shape: [{1, 2, 4, 8}, 32]   (partial-row tile -> SrcB)
//   in1 tile shape: [32, 32]             (full tile -> SrcA, any MX format the unpacker converts to the SrcA format)
//   rt_dim: 1
//   ct_dim: any integer from 1 to 16
//   kt_dim: any integer >= 1 (Blackhole needs an even kt_dim, Quasar does not)
//
// Weight stream layout in L1: tiles in row-major (k, c) order, packed back to back at their own size
// (MxInt8 1056 B, MxInt4 544 B, MxInt2 288 B); zero tiles take no space.
//
// Metadata: one 32-bit word per weight tile, in the same (k, c) order, read on the RISC-V by the unpack thread
// only (math multiplies every tile, a zero tile as a zeroed SrcA bank). The word is the UNPACR0 ARG_FIFO mode-1
// entry itself, with the address relative to the start of the weight stream:
//   [27:20] tile L1 format (DataFormat), 0 marks a zero tile
//   [19:0]  tile offset from the weight stream base, in 16 B units
// A zero tile's word is 0 as a whole, so testing for one is a single compare on the RISC-V.
// Blackhole packs 3 bits per tile and decodes them on the RISC-V. On Quasar the RISC-V has to feed every tile
// through the ARG_FIFO anyway, so storing the entry costs 4 B per tile (under 1.5% of an MxInt2 tile) and leaves a
// single add per tile on the unpack thread.

constexpr std::uint32_t COMPRESSED_MM_ENTRY_ADDR_MASK   = BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_MASK;
constexpr std::uint32_t COMPRESSED_MM_ENTRY_FORMAT_MASK = BUFFER_DESCRIPTOR_TABLE_REG0_TILE_FORMAT_MASK;

// The zero-tile marker reuses format code 0 (Float32), which is never a compressed weight format.
static_assert(static_cast<std::uint32_t>(DataFormat::Float32) == 0, "compressed_custom_mm uses format code 0 as the zero-tile marker");

/**
 * @brief Returns true when a metadata word marks an all-zero weight tile, which has no data in L1.
 *
 * @param entry: Metadata word of the tile, see the layout above.
 */
inline bool _llk_compressed_custom_mm_is_zero_tile_(const std::uint32_t entry)
{
    // A non-zero tile always has a non-zero format; a zero tile's whole word is 0 (see the layout above).
    return entry == 0;
}

/**
 * @brief Returns true when the Input0 tile shape is one compressed_custom_mm supports: [{1, 2, 4, 8}, 32].
 *
 * @param src_b_shape: Input 0/SrcB tile shape.
 */
inline bool _llk_compressed_custom_mm_is_valid_src_b_shape_(const ckernel::TensorShape src_b_shape)
{
    const bool valid_rows = src_b_shape.face_r_dim == 1 || src_b_shape.face_r_dim == 2 || src_b_shape.face_r_dim == 4 || src_b_shape.face_r_dim == 8;
    return valid_rows && src_b_shape.face_c_dim == ckernel::MAX_FACE_C_DIM && src_b_shape.num_faces_r_dim == 1 &&
           src_b_shape.num_faces_c_dim == ckernel::MAX_NUM_FACES_C_DIM;
}
