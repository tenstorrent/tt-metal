// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "tensor_shape.h"

namespace ckernel
{

// Tiles per unpack context: the data unpacker's Z counter addresses 256 faces.
constexpr std::uint32_t REDUCE_BLOCK_MAX_TILES = 32;

// Unpack and math agree on this: SUM/AVG REDUCE_ROW of full 32x32 tiles keeps the scaler in SrcA for a whole chunk.
template <PoolType pool_type, ReduceDim reduce_dim>
constexpr bool reduce_block_holds_scaler(const TensorShape& tensor_shape)
{
    return (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX) && (tensor_shape.total_num_faces() == 4) &&
           (tensor_shape.face_r_dim == FACE_R_DIM);
}

// Unpack and math must agree on the COL pair layout (two faces of a face row per SrcA bank); both inits opt in with col_pair.
template <PoolType pool_type, ReduceDim reduce_dim>
constexpr bool reduce_col_paired(const TensorShape& tensor_shape)
{
    return (reduce_dim == ReduceDim::REDUCE_COL) && (pool_type != PoolType::MAX) && (tensor_shape.total_num_faces() == 4) &&
           (tensor_shape.face_r_dim == FACE_R_DIM);
}

// Unpack and math must agree on the SCALAR SUM/AVG pair layout (the same face of two tiles in one SrcA bank, rows 0-15 and
// 16-31); the caller of both block calls passes the same decision.
template <PoolType pool_type, ReduceDim reduce_dim>
constexpr bool reduce_scalar_paired(const TensorShape& tensor_shape)
{
    return (reduce_dim == ReduceDim::REDUCE_SCALAR) && (pool_type != PoolType::MAX) && (tensor_shape.total_num_faces() == 4) &&
           (tensor_shape.face_r_dim == FACE_R_DIM);
}
} // namespace ckernel
