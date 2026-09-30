// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Buffer descriptor (BFD) types and table access.

#include <cstdint>

#include "cfg_defines.h"
#include "llk_assert.h"
#include "tensix_types.h"
#include "tensor_shape.h"

namespace ckernel::trisc
{

// Num of words in buffer descriptor struct
constexpr static std::uint32_t BD_NUM_WORDS = 3;

// Number of entries in the buffer descriptor table (physically partitioned per TRISC; ids are 0..31).
constexpr std::uint32_t BD_TABLE_NUM_ENTRIES = 32;

// Selects how L1 is addressed for an op when building its buffer descriptor.
// Continuous: normal tile layout (y_dim = face_r_dim).
// Strided: PACR/UNPACR_STRIDE tiny-tile quirk. HW/programming constraints require the buffer
//          descriptor to be programmed as y_dim = 1 so that L1 rows are indexed as tiles.
enum class L1AccessMode : std::uint8_t
{
    Continuous = 0,
    Strided    = 1,
};

// Real UNPACR/PACR hardware engines that consume buffer descriptors. One "current id" slot each;
// compile-time ownership ties each engine to a TRISC role (see bfd_engine_owned_by_trisc).
//
// Unp2_Slice0/Unp2_Slice1 are one engine, not two: a binary SrcS op needs both input descriptors
// live at once, so UNPACR2 gets two current[] slots.
enum class BfdResource : std::uint8_t
{
    Unp0 = 0,
    Unp1,
    Pack0,
    Pack1,
    Unp2_Slice0,
    Unp2_Slice1,
    Count
};

// Points to the config space
std::uint32_t volatile* const cfg = (std::uint32_t volatile*)TENSIX_CFG_BASE;
// Points to the buffer table
buffer_descriptor_u volatile* const bd_table = (buffer_descriptor_u volatile* const)(cfg + BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_ADDR32);

/**
 * @brief helper function used to compute z-dim for buf_desc from TensorShape.
 * Compares two values, then computes the square of the smaller value.
 *
 * @param input1/input2: values to be compared, then squared
 */
inline std::uint16_t compute_square_of_min(std::uint8_t input1, std::uint8_t input2)
{
    return (input1 < input2) ? input1 * input1 : input2 * input2;
}

/**
 * @brief Helper function to ensure valid tile sizes are programmed into the buffer descriptor.
 * valid tile sizes:
    1x16: x=16, y=1, z=1
    2x16: x=16, y=2, z=1
    4x16: x=16, y=4, z=1
    8x16: x=16, y=8, z=1
    16x16: x=16, y=16, z=1
    32x32: x=16, y=16, z=4
 * @param x_dim: Face column dimension
 * @param y_dim: Face row dimension after applying the L1 access mode
 * @param z_dim: Face count after applying the L1 access mode
 * @tparam MODE: L1 access mode the descriptor was built for. Strided ops (PACR/UNPACR_STRIDE
 *        tiny-tiles) must be programmed with y_dim = 1 to index L1 rows as tiles.
 */
template <L1AccessMode MODE = L1AccessMode::Continuous>
inline void validate_buffer_desc(const std::uint8_t x_dim, const std::uint8_t y_dim, const std::uint8_t z_dim)
{
    LLK_ASSERT(x_dim == 16, "x_dim must be 16");
    LLK_ASSERT(y_dim == 16 || y_dim == 8 || y_dim == 4 || y_dim == 2 || y_dim == 1, "y_dim must be powers of 2 <= 16");
    LLK_ASSERT(z_dim == 1 || z_dim == 4, "z_dim must be 1 or 4");
    if (z_dim == 4)
    {
        LLK_ASSERT(y_dim == 16, "y_dim must be 16 when z_dim is 4");
    }
    if constexpr (MODE == L1AccessMode::Strided)
    {
        LLK_ASSERT(y_dim == 1, "Strided L1 access requires buffer descriptor y_dim == 1");
    }
}

/**
 * @brief Populates buffer table entry for TDMA engines
 * @param buf_desc_id: Buffer descriptor id into the buffer descriptor table
 * @param word0: Encoded L1 base address and data format
 * @param word1: Encoded LMT base address and X dimension
 * @param word2: Encoded Y and Z dimensions
 */
inline void _configure_buf_desc_table_(const std::uint32_t buf_desc_id, const std::uint32_t word0, const std::uint32_t word1, const std::uint32_t word2)
{
    // Guards the invalid sentinel (BFD_ID_INVALID) and any OOB id from a wild write into cfg space.
    LLK_ASSERT(buf_desc_id < BD_TABLE_NUM_ENTRIES, "buf_desc_id out of range");
    bd_table[buf_desc_id].words[0] = word0;
    bd_table[buf_desc_id].words[1] = word1;
    bd_table[buf_desc_id].words[2] = word2;
}

/**
 * @brief Creates a buffer descriptor from TensorShape and other needed parameters
 * Currently supported buffer descriptor dimensions are:
 * x=16; y=[1, 2, 4, 8, 16]; z=1; or x=16; y=16; z=4; these are hardware constraints.
 *
 * @tparam MODE: L1 access mode. Strided (PACR/UNPACR_STRIDE tiny-tiles) forces y_dim = 1 and z_dim = 1
 *        so L1 rows are indexed as tiles; Continuous keeps the tensor-shape derived y_dim, z_dim.
 * @param tensor_shape: Tile/face dimensions and shape of input tensor
 * @param base_l1_16B: base address of the buffer in L1
 * @param data_format: L1 data encoding format
 */
template <L1AccessMode MODE = L1AccessMode::Continuous>
inline buffer_descriptor_u construct_buf_desc(const TensorShape& tensor_shape, unsigned base_l1_16B, unsigned data_format)
{
    const std::uint8_t x_dim = tensor_shape.face_c_dim;
    const std::uint8_t y_dim = MODE == L1AccessMode::Strided ? 1 : tensor_shape.face_r_dim;
    const std::uint8_t z_dim =
        MODE == L1AccessMode::Strided ? 1 : static_cast<std::uint8_t>(compute_square_of_min(tensor_shape.num_faces_r_dim, tensor_shape.num_faces_c_dim));
    validate_buffer_desc<MODE>(x_dim, y_dim, z_dim);

    const std::uint32_t word0 = (base_l1_16B & BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_MASK) |
                                ((data_format << BUFFER_DESCRIPTOR_TABLE_REG0_TILE_FORMAT_SHAMT) & BUFFER_DESCRIPTOR_TABLE_REG0_TILE_FORMAT_MASK);
    const std::uint32_t word1 = static_cast<std::uint32_t>(x_dim) << BUFFER_DESCRIPTOR_TABLE_REG0_TILE_X_DIM_SHAMT;
    const std::uint32_t word2 = (static_cast<std::uint32_t>(y_dim) << BUFFER_DESCRIPTOR_TABLE_REG0_TILE_Y_DIM_SHAMT) |
                                (static_cast<std::uint32_t>(z_dim) << BUFFER_DESCRIPTOR_TABLE_REG0_TILE_Z_DIM_SHAMT);
    return {{word0, word1, word2, 0}};
}

} // namespace ckernel::trisc
