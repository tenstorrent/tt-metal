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
// All APIs below generate scaler tiles consumed by the reduce LLK.
// They must ONLY be used for that purpose — not for arbitrary constant tiles.
//
// IMPORTANT — column-axis vs row-axis fill:
//   The LLK processes scaler and input faces in lockstep (face N × face N).
//   For REDUCE_ROW, the unpacker transposes each input face, so the scaler's
//   positions weight input COLUMNS via the left/right face split (face & 1).
//   For REDUCE_COL, no transpose occurs, so the scaler's positions weight
//   input ROWS via the top/bottom face split (face >= 2).
//
//   Dimension-aware overloads (taking ReduceDim) handle this automatically.
//   Column-axis-only overloads (without ReduceDim) are retained for backward
//   compatibility but are only correct for REDUCE_ROW.
//
// calculate_and_prepare_reduce_scaler (DEFAULT / PREFERRED):
//   Computes the standard reduce scaler (1/N for AVG, 1.0 for SUM/MAX) from
//   pool type, reduce dimension, and reduce factor, then writes it to a CB tile.
//   Automatically dispatches to the correct axis fill based on reduce_dim.
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
 * @brief Prepares a CB tile for reduce using a caller-provided float scaler (column-axis fill)
 *
 * Converts the float scaler to the appropriate bit representation based on
 * the circular buffer's data format, then fills row 0 of each face.
 * Data format and tile shape (half/full) are deduced from the circular buffer.
 *
 * NOTE: This overload always uses column-axis fill (left/right face split),
 * which is correct for REDUCE_ROW. For REDUCE_COL, use the dimension-aware
 * overload that takes ReduceDim.
 *
 * @tparam cb_id Circular buffer ID to write the tile to (must be constexpr)
 * @tparam tile_columns_to_fill Number of tile columns to fill (1-32, default 32 = full tile).
 *         Used when the last input tile in the reduce dimension is not full, so unused columns are left as zeros.
 * @param scaler_f Float scaler value to fill the tile with
 */
template <uint32_t cb_id, uint32_t tile_columns_to_fill = tt::constants::TILE_WIDTH>
FORCE_INLINE void prepare_reduce_scaler(float scaler_f);

/**
 * @brief Dimension-aware: prepares a CB tile for reduce with correct axis fill
 *
 * Dispatches to column-axis fill for REDUCE_ROW or row-axis fill for REDUCE_COL
 * based on reduce_dim. For full tiles (positions_to_fill == TILE_WIDTH), both
 * axes produce identical results.
 *
 * @tparam cb_id Circular buffer ID to write the tile to (must be constexpr)
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW or REDUCE_COL)
 * @tparam positions_to_fill Number of positions along the reduce dimension to fill (1-32, default 32).
 * @param scaler_f Float scaler value to fill the tile with
 */
template <uint32_t cb_id, ReduceDim reduce_dim, uint32_t positions_to_fill = tt::constants::TILE_WIDTH>
FORCE_INLINE void prepare_reduce_scaler(float scaler_f);

/**
 * @brief Generate a reduce scaler tile with format and tile shape deduced from cb_id
 *
 * Computes the appropriate scaler value based on pool type, reduce dimension,
 * and reduce factor. Supports both bfloat16 and float32 formats.
 * Data format and tile shape (half/full) are deduced from the circular buffer.
 * Automatically dispatches to the correct axis fill based on reduce_dim.
 *
 * For AVG pooling with REDUCE_SCALAR, uses 1/sqrt(N) since the LLK applies the
 * scaler twice (row then col). For AVG with REDUCE_ROW/REDUCE_COL, uses 1/N.
 * For SUM/MAX, the reduce_factor is ignored and the scaler is 1.0.
 *
 * @tparam cb_id Circular buffer ID to write the tile to (must be constexpr)
 * @tparam pool_type Type of pooling operation (SUM, AVG, MAX)
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 * @tparam positions_to_fill Number of positions along the reduce dimension (1-32, default 32).
 * @tparam reduce_factor Number of elements being reduced (N). Must be set for AVG; not used for MAX and SUM.
 */
template <
    uint32_t cb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t positions_to_fill = tt::constants::TILE_WIDTH,
    uint32_t reduce_factor = 1>
FORCE_INLINE void calculate_and_prepare_reduce_scaler();

// =============================================================================
// Partial scaler helpers - generate BOTH full and partial scaler tiles
//
// These convenience functions generate two scaler tiles for the partial-tile
// reduction pattern. They push tile 0 (full, all 32 positions filled) then
// tile 1 (partial, only partial_positions filled) into the CB.
//
// Use with compute_kernel_lib::ReducePartialScaler::last_tile_at(1) on the
// compute side to select tile 1 for the last tile during reduction.
//
// IMPORTANT: The scaler CB must have capacity for at least 2 tiles.
//
// See ReducePartialScaler documentation in reduce_helpers_compute.hpp for
// the full pattern description and usage examples.
// =============================================================================

/**
 * @brief Generate two scaler tiles (full + partial) for REDUCE_ROW (column-axis fill)
 *
 * Pushes two tiles into the scaler CB:
 *   Tile 0: scaler_f in all TILE_WIDTH columns (for full interior tiles)
 *   Tile 1: scaler_f in only partial_tile_columns columns (for last W tile)
 *
 * NOTE: Column-axis fill only — correct for REDUCE_ROW. For REDUCE_COL or
 * dimension-agnostic code, use the overload that takes ReduceDim.
 *
 * @tparam cb_id Circular buffer ID to write the tiles to (must be constexpr)
 * @tparam partial_tile_columns Number of valid columns in the last tile.
 *         Must be in range [1, TILE_WIDTH-1]. Computed as origin_W % TILE_WIDTH.
 * @param scaler_f Float scaler value (e.g., 1.0 for SUM, 1/N for AVG)
 */
template <uint32_t cb_id, uint32_t partial_tile_columns>
FORCE_INLINE void prepare_partial_reduce_scalers(float scaler_f);

/**
 * @brief Dimension-aware: generate two scaler tiles (full + partial)
 *
 * Dispatches to column-axis fill for REDUCE_ROW or row-axis fill for REDUCE_COL.
 *
 * @tparam cb_id Circular buffer ID to write the tiles to (must be constexpr)
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW or REDUCE_COL)
 * @tparam partial_positions Number of valid positions in the last tile along the reduce dim.
 *         Must be in range [1, 31]. Computed as origin_dim % TILE_DIM.
 * @param scaler_f Float scaler value (e.g., 1.0 for SUM, 1/N for AVG)
 */
template <uint32_t cb_id, ReduceDim reduce_dim, uint32_t partial_positions>
FORCE_INLINE void prepare_partial_reduce_scalers(float scaler_f);

/**
 * @brief Dimension-aware: generate two scaler tiles (full + partial) with auto-computed scaler
 *
 * Computes the scaler from pool_type/reduce_dim/reduce_factor, then pushes two tiles.
 * Dispatches to the correct axis fill based on reduce_dim.
 *
 * @tparam cb_id Circular buffer ID to write the tiles to (must be constexpr)
 * @tparam pool_type Type of pooling operation (SUM, AVG, MAX)
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW, REDUCE_COL). REDUCE_SCALAR not supported.
 * @tparam partial_positions Number of valid positions in the last tile (origin_dim % TILE_DIM)
 * @tparam reduce_factor Number of elements being reduced (N). Required for AVG.
 */
template <
    uint32_t cb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t partial_positions,
    uint32_t reduce_factor = 1>
FORCE_INLINE void calculate_and_prepare_partial_reduce_scalers();

// =============================================================================
// Row-axis scaler helpers (for REDUCE_COL)
//
// REDUCE_COL natively computes scaler @ input (matmul). The scaler positions
// weight input ROWS, mapped via the top/bottom face split:
//   Faces 0,1 (top)    → input rows 0-15
//   Faces 2,3 (bottom) → input rows 16-31
//
// These functions fill based on this row axis, unlike the column-axis functions
// above which fill based on the left/right face split.
//
// Prefer the dimension-aware overloads of prepare_reduce_scaler and
// prepare_partial_reduce_scalers which dispatch automatically.
// =============================================================================

/**
 * @brief Prepare a single reduce scaler tile with row-axis partial fill
 *
 * @tparam cb_id Circular buffer ID
 * @tparam tile_rows_to_fill Number of row positions to fill (1-32, default 32)
 * @param scaler_f Float scaler value
 */
template <uint32_t cb_id, uint32_t tile_rows_to_fill = tt::constants::TILE_HEIGHT>
FORCE_INLINE void prepare_reduce_scaler_rows(float scaler_f);

/**
 * @brief Generate two scaler tiles (full + partial rows) for REDUCE_COL
 *
 * Pushes two tiles:
 *   Tile 0: full scaler (all 32 row positions filled)
 *   Tile 1: partial scaler (only tile_rows positions filled, based on top/bottom face split)
 *
 * @tparam cb_id Circular buffer ID
 * @tparam partial_tile_rows Number of valid rows in the last H tile (origin_H % TILE_HEIGHT)
 * @param scaler_f Float scaler value
 */
template <uint32_t cb_id, uint32_t partial_tile_rows>
FORCE_INLINE void prepare_partial_reduce_scalers_rows(float scaler_f);

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
