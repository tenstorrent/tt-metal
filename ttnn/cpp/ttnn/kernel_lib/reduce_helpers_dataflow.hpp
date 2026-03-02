// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include <tt-metalium/constants.hpp>

namespace dataflow_kernel_lib {

using ckernel::PoolType;
using ckernel::ReduceDim;

// =============================================================================
// Reduce scaler helpers API
//
// calculate_and_prepare_reduce_scaler: Automatically computes the correct scaler based on
//   pool type, reduce dimension, and reduce factor (e.g. 1/N for AVG, 1.0 for
//   SUM/MAX). Use this when performing standard reduce operations.
//
// prepare_reduce_scaler: Prepares a CB tile for reduce using a caller-provided
//   float scaler. Use this when you need a custom scaler value that doesn't
//   follow the standard reduce conventions. Accepts a runtime float value.
// =============================================================================

/**
 * @brief Prepares a CB tile for reduce using a caller-provided float scaler
 *
 * Converts the float scaler to the appropriate bit representation based on
 * the circular buffer's data format, then fills row 0 of each face.
 * Data format and tile shape (half/full) are deduced from the circular buffer.
 *
 * @tparam cb_id Circular buffer ID to write the tile to (must be constexpr)
 * @tparam tile_columns_to_fill Number of tile columns to fill (1-32, default 32 = full tile).
 *         When < 32, only the first tile_columns_to_fill columns of row 0 are filled (for partial last tiles).
 * @param scaler_f Float scaler value to fill the tile with
 */
template <uint32_t cb_id, uint32_t tile_columns_to_fill = tt::constants::TILE_WIDTH>
FORCE_INLINE void prepare_reduce_scaler(float scaler_f);

/**
 * @brief Generate a reduce scaler tile with format and tile shape deduced from cb_id
 *
 * Computes the appropriate scaler value based on pool type, reduce dimension,
 * and reduce factor. Supports both bfloat16 and float32 formats.
 * Data format and tile shape (half/full) are deduced from the circular buffer.
 *
 * For AVG pooling with REDUCE_SCALAR, uses 1/sqrt(N) since the LLK applies the
 * scaler twice (row then col). For AVG with REDUCE_ROW/REDUCE_COL, uses 1/N.
 * For SUM/MAX, the reduce_factor is ignored and the scaler is 1.0.
 *
 * @tparam cb_id Circular buffer ID to write the tile to (must be constexpr)
 * @tparam pool_type Type of pooling operation (SUM, AVG, MAX)
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 * @tparam tile_columns_to_fill Number of tile columns to fill (1-32, default 32 = full tile).
 *         When < 32, only the first tile_columns_to_fill columns of row 0 are filled (for partial last tiles).
 * @tparam reduce_factor Number of elements being reduced (N). Must be non-zero for AVG.
 */
template <
    uint32_t cb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t tile_columns_to_fill = tt::constants::TILE_WIDTH,
    uint32_t reduce_factor = 1>
FORCE_INLINE void calculate_and_prepare_reduce_scaler();

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
