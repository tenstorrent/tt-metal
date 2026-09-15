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
enum class BfdResource : std::uint8_t
{
    Unp0 = 0,
    Unp1,
    Pack0,
    Pack1,
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
 * @param buf_desc: Contains L1 buffer descriptor information
 * @tparam MODE: L1 access mode the descriptor was built for. Strided ops (PACR/UNPACR_STRIDE
 *        tiny-tiles) must be programmed with y_dim = 1 to index L1 rows as tiles.
 */
template <L1AccessMode MODE = L1AccessMode::Continuous>
inline void validate_buffer_desc(const buffer_descriptor_u& buf_desc)
{
    LLK_ASSERT(buf_desc.f.x_dim == 16, "x_dim must be 16");
    LLK_ASSERT(
        buf_desc.f.y_dim == 16 || buf_desc.f.y_dim == 8 || buf_desc.f.y_dim == 4 || buf_desc.f.y_dim == 2 || buf_desc.f.y_dim == 1,
        "y_dim must be powers of 2 <= 16");
    LLK_ASSERT(buf_desc.f.z_dim == 1 || buf_desc.f.z_dim == 4, "z_dim must be 1 or 4");
    if (buf_desc.f.z_dim == 4)
    {
        LLK_ASSERT(buf_desc.f.y_dim == 16, "y_dim must be 16 when z_dim is 4");
    }
    if constexpr (MODE == L1AccessMode::Strided)
    {
        LLK_ASSERT(buf_desc.f.y_dim == 1, "Strided L1 access requires buffer descriptor y_dim == 1");
    }
}

/**
 * @brief Populates buffer table entry for TDMA engines
 * @param buf_desc_id: Buffer descriptor id into the buffer descriptor table
 * @param buf_desc: Contains L1 buffer descriptor information
 */
inline void _configure_buf_desc_table_(const std::uint32_t buf_desc_id, const buffer_descriptor_u& buf_desc)
{
    // Guards the invalid sentinel (BFD_ID_INVALID) and any OOB id from a wild write into cfg space.
    LLK_ASSERT(buf_desc_id < BD_TABLE_NUM_ENTRIES, "buf_desc_id out of range");
    for (std::uint32_t i = 0; i < BD_NUM_WORDS; i++)
    {
        bd_table[buf_desc_id].words[i] = buf_desc.words[i];
    }
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
    buffer_descriptor_u buf_desc = {0};
    buf_desc.f.x_dim             = tensor_shape.face_c_dim;
    buf_desc.f.y_dim             = tensor_shape.face_r_dim;
    if (tensor_shape.num_faces_r_dim == tensor_shape.num_faces_c_dim)
    {
        buf_desc.f.z_dim = tensor_shape.total_num_faces();
    }
    else
    {
        buf_desc.f.z_dim = static_cast<std::uint8_t>(compute_square_of_min(tensor_shape.num_faces_r_dim, tensor_shape.num_faces_c_dim));
    }
    buf_desc.f.l1_addr_16B = base_l1_16B;
    buf_desc.f.format      = static_cast<std::uint8_t>(data_format);

    if constexpr (MODE == L1AccessMode::Strided)
    {
        // PACR_STRIDE quirk: program BD as 1x1x16 so L1 addressing indexes rows as tiles.
        buf_desc.f.y_dim = 1;
        buf_desc.f.z_dim = 1;
    }

    validate_buffer_desc<MODE>(buf_desc);

    return buf_desc;
}

} // namespace ckernel::trisc
