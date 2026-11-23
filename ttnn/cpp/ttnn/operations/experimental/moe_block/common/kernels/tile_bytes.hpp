// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The byte sizes of one 32 x 32 tile in the data formats the MoE block's ops move, shared by the host and the device
// kernels so both sides use the same names. Built on tt::constants::TILE_HW: tt-metalium/constants.hpp on the host,
// its device-side counterpart api/tile_constants.h in kernels (the two may not include each other).
#pragma once

#include <cstdint>

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/tile_constants.h"
#else
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#endif

namespace moe_block_tiles {

inline constexpr uint32_t kTileElements = tt::constants::TILE_HW;             // 32 x 32
inline constexpr uint32_t kBf16TileBytes = kTileElements * sizeof(uint16_t);  // a bfloat16 tile: 2048 B
inline constexpr uint32_t kFp32TileBytes = kTileElements * sizeof(uint32_t);  // a float32 (or uint32) tile: 4096 B

#if !defined(KERNEL_BUILD) && !defined(FW_BUILD)
static_assert(kBf16TileBytes == tt::tile_size(tt::DataFormat::Float16_b));
static_assert(kFp32TileBytes == tt::tile_size(tt::DataFormat::Float32));
#endif

}  // namespace moe_block_tiles
