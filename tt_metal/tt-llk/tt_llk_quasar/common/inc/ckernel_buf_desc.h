// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Buffer descriptor (BFD) types and table access.

#include <cfg_defines.h>

#include <cstdint>

#include "ckernel_trisc_id.h"
#include "llk_assert.h"
#include "tensix_types.h"
#include "tensor_shape.h"

namespace ckernel::trisc
{

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

// Buffer descriptor (BFD) ids: one fixed table entry per TDMA engine, re-programmed by every op
// init and baked into the MOP by id. BFD ids are an LLK-internal resource: nothing above the LLK API
// layer may hold or pass one. The ids sit inside each TRISC's documented partition of the table,
// so threads never write each other's entries:
//
//   unpack thread -> ids [0, 16):  UNP0 = 0, UNP1 = 1
//   math thread   -> none (math issues no TDMA)
//   pack thread   -> ids [16, 24): PACK0 = 16
//   SFPU thread   -> ids [24, 32): PACK1 = 24, UNP2_SLICE0 = 25, UNP2_SLICE1 = 26
//
// UNP2_SLICE0/UNP2_SLICE1 are one engine (UNPACR2), not two: a binary SrcS op needs both input
// descriptors live at once.
//
// REQUIREMENT: correctness relies on Tensix TTSync global-CFG tracking (TRACK_GLOBAL_CFG, see
// set_ttsync_enables in ckernel.h) being ENABLED on the unpack, pack and SFPU threads (it is on at
// reset). It makes the TRISC store that re-programs a BD entry wait for any in-flight UNPACR/PACR
// still using that entry. With it disabled, a re-init can overwrite an entry under an in-flight op
// and the op runs with a stale descriptor (corrupt data).
constexpr std::uint8_t BFD_ID_UNP0        = 0;
constexpr std::uint8_t BFD_ID_UNP1        = 1;
constexpr std::uint8_t BFD_ID_PACK0       = 16;
constexpr std::uint8_t BFD_ID_PACK1       = 24;
constexpr std::uint8_t BFD_ID_UNP2_SLICE0 = 25;
constexpr std::uint8_t BFD_ID_UNP2_SLICE1 = 26;

// Points to the config space
std::uint32_t volatile* const cfg = reinterpret_cast<std::uint32_t volatile*>(TENSIX_CFG_BASE);
// Points to the buffer table
buffer_descriptor_u volatile* const bd_table = reinterpret_cast<buffer_descriptor_u volatile*>(cfg + BUFFER_DESCRIPTOR_TABLE_REG0_L1_BASE_ADDR_ADDR32);

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
 * @param z_dim: Faces per HW tile (4 for a 2x2 face grid, else 1); forced to 1 for Strided mode
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
 * @param word1: Encoded L1 limit address (left 0 by construct_buf_desc) and X dimension
 * @param word2: Encoded Y and Z dimensions
 * @note Each engine re-programs one fixed entry (BFD_ID_* above). Safe only while TTSync
 *       TRACK_GLOBAL_CFG is enabled on this thread: the store then waits for in-flight UNPACR/PACR
 *       using the entry (see the REQUIREMENT note at BFD_ID_UNP0).
 */
inline void _configure_buf_desc_table_(const std::uint32_t buf_desc_id, const std::uint32_t word0, const std::uint32_t word1, const std::uint32_t word2)
{
    // Guards any OOB id from a wild write into cfg space.
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

/**
 * @brief Program the fixed table entry BFD_ID of an engine from the L1 buffer info.
 * @tparam BFD_ID: fixed id of the engine (BFD_ID_*); must belong to the compiling thread's partition
 * @tparam MODE: L1 access mode (Continuous, or Strided for PACR/UNPACR_STRIDE tiny-tiles)
 * @param tensor_shape: Tile/face dimensions and shape of the buffer
 * @param l1_base_addr: base address of the buffer in L1 (16B units)
 * @param l1_data_format: L1 data encoding format
 */
template <std::uint8_t BFD_ID, L1AccessMode MODE = L1AccessMode::Continuous>
inline void program_bfd(const TensorShape& tensor_shape, const std::uint32_t l1_base_addr, const std::uint32_t l1_data_format)
{
    // Cross-thread table writes are not ordered by TTSync, so each thread may only write its own partition.
    static_assert(
        (ckernel::TRISC_ID == 0 && BFD_ID <= BFD_ID_UNP1) || (ckernel::TRISC_ID == 2 && BFD_ID == BFD_ID_PACK0) ||
            (ckernel::TRISC_ID == 3 && BFD_ID >= BFD_ID_PACK1 && BFD_ID <= BFD_ID_UNP2_SLICE1),
        "BFD id not owned by the compiling thread (math owns none)");
    const auto buf_desc = construct_buf_desc<MODE>(tensor_shape, l1_base_addr, l1_data_format);
    _configure_buf_desc_table_(BFD_ID, buf_desc.words[0], buf_desc.words[1], buf_desc.words[2]);
}

} // namespace ckernel::trisc
