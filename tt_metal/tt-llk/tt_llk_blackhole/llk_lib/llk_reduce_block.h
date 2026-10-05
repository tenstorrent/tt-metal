// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "tensor_shape.h"

namespace ckernel
{

// Tiles per unpack context of a block reduce: the data unpacker's Z counter addresses 256 faces.
constexpr std::uint32_t REDUCE_BLOCK_MAX_TILES = 32;

// The unpack and math sides of a block reduce agree on this: SUM/AVG REDUCE_ROW of full 32x32 tiles keeps the scaler
// tile in SrcA for a whole chunk of the block, so it is unpacked once and released once per chunk.
template <PoolType pool_type, ReduceDim reduce_dim>
constexpr bool reduce_block_holds_scaler(const TensorShape& tensor_shape)
{
    return (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX) && (tensor_shape.total_num_faces() == 4) &&
           (tensor_shape.face_r_dim == FACE_R_DIM);
}

} // namespace ckernel
