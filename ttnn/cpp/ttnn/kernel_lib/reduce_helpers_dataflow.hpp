// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp"

namespace dataflow_kernel_lib {

using ckernel::PoolType;
using ckernel::ReduceDim;

// Default reduce factor for SUM and MAX pool types (scaler is always 1.0).
// Named constant for SUM and MAX where reduce_factor is unused.
constexpr uint32_t SUM_AND_MAX_REDUCE_FACTOR = 1;

// Scaler tiles for the reduce LLK only, not for arbitrary constant tiles. Prefer
// calculate_and_prepare_reduce_scaler (1/N for AVG, 1.0 otherwise); use prepare_reduce_scaler only for a
// non-standard value, e.g. per-core 1/N under uneven sharding.

/**
 * @brief Push one scaler tile holding scaler_f, in the row-0 layout the reduce LLK reads.
 *
 * Format and tile shape are deduced from dfb_id. valid_reduce_dim_elements_in_tile (1-32) zeroes the
 * positions past the valid elements of a partial last tile.
 */
template <uint32_t dfb_id, PoolType pool_type, ReduceDim reduce_dim>
FORCE_INLINE void prepare_reduce_scaler(
    float scaler_f, uint32_t valid_reduce_dim_elements_in_tile = tt::constants::TILE_WIDTH);

/**
 * @brief Push the standard scaler tile: 1/N for AVG (1/sqrt(N) for REDUCE_SCALAR, which applies it twice),
 * 1.0 for SUM/MAX. reduce_factor is N; valid_reduce_dim_elements_in_tile works as in prepare_reduce_scaler.
 */
template <uint32_t dfb_id, PoolType pool_type, ReduceDim reduce_dim, uint32_t reduce_factor = SUM_AND_MAX_REDUCE_FACTOR>
FORCE_INLINE void calculate_and_prepare_reduce_scaler(
    uint32_t valid_reduce_dim_elements_in_tile = tt::constants::TILE_WIDTH);

/**
 * @brief Fill and push one auxiliary tile for compute_kernel_lib::reduce().
 *
 * The pattern carries no reduction semantics; the caller pushes the tiles the reduce() call reads:
 * - ReduceAlgorithm::Native: [full scaler], or [full scaler, partial scaler] with ReducePartialMode::Scaler.
 *   The full scaler is FirstRow over the whole width; a partial scaler is FirstRow (REDUCE_ROW) or
 *   FirstRowPerFaceRow (REDUCE_COL) over the valid elements of the last reduce-dim tile.
 * - ReduceAlgorithm::TileSum: [zero], or [mask, zero] with ReducePartialMode::Mask. The mask is FirstRow
 *   (REDUCE_ROW) or FirstColumn (REDUCE_COL) holding 1.0.
 *
 * FirstRow fills row 0 and FirstColumn fills column 0 over valid_elements; FirstRowPerFaceRow fills row 0 of each
 * face over the valid elements in that face row. A full FirstRow scaler writes only the rows reduction reads;
 * every other pattern zeroes the rest of the tile. value_bits is the fill value as float32 bits. Float16_b and
 * Float32 only.
 */
template <
    uint32_t dfb_id,
    ttnn::kernel_lib::ReduceAuxiliaryTileType tile_type,
    uint32_t valid_elements = tt::constants::TILE_WIDTH,
    uint32_t value_bits = 0x3F800000>
FORCE_INLINE void prepare_reduce_auxiliary_tile();

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
