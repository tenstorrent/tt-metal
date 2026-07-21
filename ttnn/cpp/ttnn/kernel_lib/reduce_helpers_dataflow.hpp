// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>

namespace dataflow_kernel_lib {

using ckernel::PoolType;
using ckernel::ReduceDim;

// Default reduce factor for SUM and MAX pool types (scaler is always 1.0).
// Named constant for SUM and MAX where reduce_factor is unused.
constexpr uint32_t SUM_AND_MAX_REDUCE_FACTOR = 1;

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
 * @brief Prepares a DFB entry for reduce using a caller-provided float scaler
 *
 * Converts the float scaler to the appropriate bit representation based on
 * the DataflowBuffer's data format, then fills the tile with the scaler in
 * the row-0 layout required by the reduce LLK.
 *
 * Data format and tile shape (half/full) are deduced from the DataflowBuffer.
 *
 * @tparam dfb_id DataflowBuffer ID to write the entry to (must be constexpr)
 * @tparam pool_type Type of pooling operation (SUM, AVG, MAX)
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 * @param scaler_f Float scaler value to fill the entry with
 * @param valid_reduce_dim_elements_in_tile Number of valid elements along the reduce dimension
 *        in the tile (1-32, default 32 = full tile). When the last tile along the reduce
 *        dimension is partially filled, this specifies how many row or column elements contain
 *        valid data; the remaining positions are zeroed out so they do not affect the result.
 */
template <uint32_t dfb_id, PoolType pool_type, ReduceDim reduce_dim>
FORCE_INLINE void prepare_reduce_scaler(
    float scaler_f, uint32_t valid_reduce_dim_elements_in_tile = tt::constants::TILE_WIDTH);

/**
 * @brief Fill a 0/1 MASK tile for the AccumulateViaAdd partial (non-tile-aligned) reduce path.
 *
 * Writes 1.0 in the first `valid_elems` positions along the reduce dimension and 0 elsewhere, in the
 * broadcast layout the compute consumes on the last tile: REDUCE_ROW -> row-0 (mul_tiles_bcast_rows),
 * REDUCE_COL -> col-0 (mul_tiles_bcast_cols). REDUCE_SCALAR is rejected (2-D partial unsupported).
 *
 * @tparam dfb_id DataflowBuffer ID to write the mask to (must be constexpr)
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW or REDUCE_COL)
 * @param valid_elems Valid reduce-dim elements in the last tile, in [1, tile reduce-axis dim - 1].
 */
template <uint32_t dfb_id, ReduceDim reduce_dim>
FORCE_INLINE void prepare_reduce_mask(uint32_t valid_elems);

/**
 * @brief Generate a reduce scaler tile with format and tile shape deduced from dfb_id
 *
 * Computes the appropriate scaler value based on pool type, reduce dimension,
 * and reduce factor. Supports both bfloat16 and float32 formats.
 * Data format and tile shape (half/full) are deduced from the DataflowBuffer.
 *
 * For AVG pooling with REDUCE_SCALAR, uses 1/sqrt(N) since the LLK applies the
 * scaler twice (row then col). For AVG with REDUCE_ROW/REDUCE_COL, uses 1/N.
 * For SUM/MAX, the reduce_factor is ignored and the scaler is 1.0.
 *
 * @tparam dfb_id DataflowBuffer ID to write the entry to (must be constexpr)
 * @tparam pool_type Type of pooling operation (SUM, AVG, MAX)
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 * @tparam reduce_factor Number of elements being reduced (N). Must be set for AVG;
 *         use SUM_AND_MAX_REDUCE_FACTOR (default) for SUM and MAX.
 * @param valid_reduce_dim_elements_in_tile Number of valid elements along the reduce dimension
 *        in the tile (1-32, default 32 = full tile). When the last tile along the reduce
 *        dimension is partially filled, this specifies how many row or column elements contain
 *        valid data; the remaining positions are zeroed out so they do not affect the result.
 */
template <uint32_t dfb_id, PoolType pool_type, ReduceDim reduce_dim, uint32_t reduce_factor = SUM_AND_MAX_REDUCE_FACTOR>
FORCE_INLINE void calculate_and_prepare_reduce_scaler(
    uint32_t valid_reduce_dim_elements_in_tile = tt::constants::TILE_WIDTH);

// =============================================================================
// Partial-scaler convenience: emit a full + partial scaler tile pair
//
// When the reduce dimension is not tile-aligned, the compute kernel needs the
// full scaler for all but the last reduce-dim iteration, and a partial scaler
// (only `partial_positions` elements filled, the rest zeroed) for the last.
//
// These helpers emit two scaler tiles into the CB in that order:
//   tile 0 → full scaler
//   tile 1 → partial scaler
//
// Pair with compute_kernel_lib::ReducePartialScaler::last_tile_at(1) on the
// compute side. REDUCE_SCALAR is not supported (scaler is applied twice).
// =============================================================================

/**
 * @brief Emit two scaler tiles for non-tile-aligned reduce dimensions
 *
 * Writes a full-fill scaler tile followed by a partial-fill scaler tile to
 * `cb_id`. Each tile is computed with the same caller-provided float scaler;
 * only the number of valid positions differs.
 *
 * @tparam cb_id Circular buffer ID to write the tiles to (must be constexpr)
 * @tparam pool_type Type of pooling operation (SUM, AVG, MAX)
 * @tparam reduce_dim Reduction dimension. REDUCE_SCALAR is rejected at compile time.
 * @tparam partial_positions Number of valid elements along the reduce axis in the
 *         partial tile. Must be in [1, tile_dim - 1]; if you'd pass tile_dim use
 *         prepare_reduce_scaler with the single-tile path instead.
 * @param scaler_f Float scaler value to fill both tiles with
 */
template <uint32_t cb_id, PoolType pool_type, ReduceDim reduce_dim, uint32_t partial_positions>
FORCE_INLINE void prepare_partial_reduce_scalers(float scaler_f);

/**
 * @brief calculate-and-fill variant of prepare_partial_reduce_scalers
 *
 * Computes the standard reduce scaler from pool type, reduce dimension, and
 * reduce factor (1/N for AVG REDUCE_ROW/REDUCE_COL; 1.0 for SUM/MAX), then
 * emits the full + partial scaler tile pair via prepare_partial_reduce_scalers.
 *
 * @tparam cb_id Circular buffer ID
 * @tparam pool_type Pool type
 * @tparam reduce_dim Reduction dimension (REDUCE_SCALAR is rejected)
 * @tparam partial_positions Valid elements in the partial tile, in [1, tile_dim - 1]
 * @tparam reduce_factor Number of elements being reduced (used by AVG only)
 */
template <
    uint32_t cb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t partial_positions,
    uint32_t reduce_factor = SUM_AND_MAX_REDUCE_FACTOR>
FORCE_INLINE void calculate_and_prepare_partial_reduce_scalers();

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
