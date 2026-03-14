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
// Both APIs below generate a scaler tile consumed by the reduce LLK.
// They must ONLY be used for that purpose — not for arbitrary constant tiles.
//
// calculate_and_prepare_reduce_scaler (DEFAULT / PREFERRED):
//   Computes the standard reduce scaler (1/N for AVG, 1.0 for SUM/MAX) from
//   pool type, reduce dimension, and reduce factor, then writes it to a CB tile.
//   Use this for all reduce operations that use a standard scaler.
//
// prepare_reduce_scaler:
//   Writes a caller-provided float value into a CB tile for reduce.
//   Use ONLY when the reduce scaler is non-standard — i.e., it is NOT the
//   usual 1/N for AVG or 1.0 for SUM/MAX. For example:
//     - Different cores reduce over different-sized partitions (sharded with
//       uneven splits), so each core needs a different 1/N value.
//     - The scaler combines reduction with another factor (e.g., 1/(N*M)).
// =============================================================================

/**
 * @brief Prepares a CB tile for reduce using a caller-provided float scaler
 *
 * Converts the float scaler to the appropriate bit representation based on
 * the circular buffer's data format, then fills row 0 of each face.
 * Data format and tile shape (half/full) are deduced from the circular buffer.
 *
 * @tparam cb_id Circular buffer ID to write the tile to (must be constexpr)
 * @tparam valid_reduce_dim_elements_in_tile Number of valid elements along the reduce dimension
 *         in the tile (1-32, default 32 = full tile). When the last tile along the reduce
 *         dimension is partially filled, this specifies how many row or column elements contain
 *         valid data; the remaining positions are zeroed out so they do not affect the result.
 * @param scaler_f Float scaler value to fill the tile with
 */
template <uint32_t cb_id, uint32_t valid_reduce_dim_elements_in_tile = tt::constants::TILE_WIDTH>
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
 * @tparam valid_reduce_dim_elements_in_tile Number of valid elements along the reduce dimension
 *         in the tile (1-32, default 32 = full tile). When the last tile along the reduce
 *         dimension is partially filled, this specifies how many row or column elements contain
 *         valid data; the remaining positions are zeroed out so they do not affect the result.
 * @tparam reduce_factor Number of elements being reduced (N). Must be set for AVG; not used for MAX and SUM.
 */
template <
    uint32_t cb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t valid_reduce_dim_elements_in_tile = tt::constants::TILE_WIDTH,
    uint32_t reduce_factor = 1>
FORCE_INLINE void calculate_and_prepare_reduce_scaler();

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
