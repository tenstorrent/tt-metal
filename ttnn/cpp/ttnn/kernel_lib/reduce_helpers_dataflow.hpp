// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>

namespace dataflow_kernel_lib {

using ckernel::PoolType;
using ckernel::ReduceDim;

// Default reduce factor for SUM and MAX pool types (scaler is always 1.0).
// Named constant to use when you need to pass reduce_factor explicitly to reach compute_uses_reduce_tile.
constexpr uint32_t SUM_AND_MAX_REDUCE_FACTOR = 1;

// =============================================================================
// Partial-scaler descriptor (type-dispatched, mirrors compute_kernel_lib's
// NoAccumulation / Accumulate pattern and pairs with its ReducePartialScaler enum)
//
// Selects the scaler tiles emitted for compute_kernel_lib::reduce<>: NoPartial
// (default) emits one full tile (ReducePartialScaler::None); PartialLastTile
// emits two tiles — full at index 0, partial at index 1 — for a non-tile-aligned
// reduce axis (ReducePartialScaler::LastTile uses the partial tile for the last
// reduce-dim tile only; the scaler DFB must hold 2 tiles).
// Partial = only the first `valid_reduce_dim_elements` positions along the
// reduce axis are filled; the rest are zeroed so they do not contribute.
//
// There is also PartialOnlyTile (one partial tile) for special cases where the
// whole reduce axis is known to fit in a single, partially-valid tile. This is
// for compute kernels calling the reduce LLK directly — the compute reduce
// helper itself only consumes the NoPartial / PartialLastTile layouts.
// =============================================================================

// Empty marker type → one full tile. Compiles away like NoAccumulation.
struct NoPartial {};

// One partial tile carrying the count of valid reduce-axis positions.
struct PartialOnlyTile {
    uint32_t valid_reduce_dim_elements = 0;
    static constexpr PartialOnlyTile with_valid_reduce_dim_elements(uint32_t n) { return {n}; }
};

// Full tile (index 0) + partial tile (index 1) with the given valid positions.
struct PartialLastTile {
    uint32_t valid_reduce_dim_elements = 0;
    static constexpr PartialLastTile with_valid_reduce_dim_elements(uint32_t n) { return {n}; }
};

template <typename T>
inline constexpr bool is_partial_scaler_v =
    std::is_same_v<T, NoPartial> || std::is_same_v<T, PartialOnlyTile> || std::is_same_v<T, PartialLastTile>;

// =============================================================================
// Reduce scaler helpers API
//
// Both APIs below generate the scaler tiles consumed by the reduce LLK — one
// tile by default, two (full + partial) with PartialLastTile.
// They must ONLY be used for that purpose — not for arbitrary constant tiles.
//
// calculate_and_prepare_reduce_scaler (DEFAULT / PREFERRED):
//   Computes the standard reduce scaler (1/N for AVG, 1.0 for SUM/MAX) from
//   pool type, reduce dimension, and reduce factor, then writes the tile(s).
//   Use this for all reduce operations that use a standard scaler.
//
// prepare_reduce_scaler:
//   Writes a caller-provided float value into the scaler tile(s) for reduce.
//   Use ONLY when the reduce scaler is non-standard — i.e., it is NOT the
//   usual 1/N for AVG or 1.0 for SUM/MAX. For example:
//     - Different cores reduce over different-sized partitions (sharded with
//       uneven splits), so each core needs a different 1/N value.
//     - The scaler combines reduction with another factor (e.g., 1/(N*M)).
// =============================================================================

/**
 * @brief Prepares one or two DFB entries (per partial_scaler) for reduce using a caller-provided float scaler
 *
 * Converts the float scaler to the appropriate bit representation based on
 * the DataflowBuffer's data format, then fills each tile with the scaler in
 * the layout required by the reduction:
 *   - Row-0 fill (reduce LLK path): used for REDUCE_COL, REDUCE_SCALAR, and MAX
 *   - Col-0 fill (matmul path): used for REDUCE_ROW with SUM or AVG
 *
 * Data format and tile shape (half/full) are deduced from the DataflowBuffer.
 *
 * @tparam dfb_id DataflowBuffer ID to write the entry to (must be constexpr)
 * @tparam pool_type Type of pooling operation (SUM, AVG, MAX). Default MAX selects row-0 fill.
 * @tparam reduce_dim Reduction dimension (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR).
 *         Default REDUCE_COL selects row-0 fill.
 * @tparam compute_uses_reduce_tile When true, forces row-0 fill (reduce LLK layout) even for
 *         SUM/AVG + REDUCE_ROW combinations that would normally use col-0 fill (matmul layout).
 *         Set to true when the compute kernel uses reduce_tile LLK directly instead of
 *         compute_kernel_lib::reduce (which auto-switches to matmul for REDUCE_ROW SUM/AVG).
 * @tparam PartialScalerT Partial-scaler descriptor type, deduced from `partial_scaler`.
 *         NoPartial (default) emits one full tile; PartialLastTile emits a full + partial
 *         tile pair for a non-tile-aligned reduce axis. In special cases where the compute
 *         side does not use the reduce helper, a single partial tile may be all that is
 *         needed — use PartialOnlyTile for that. REDUCE_SCALAR requires NoPartial.
 * @param scaler_f Float scaler value to fill the entry with
 * @param partial_scaler Partial-scaler descriptor (default NoPartial{} = one full tile).
 *        Pass PartialLastTile::with_valid_reduce_dim_elements(n) for a non-tile-aligned
 *        reduce axis (full tile first, then partial; the scaler DFB must hold 2 tiles).
 *        In special cases where the compute side does not use the reduce helper, a single
 *        partial tile may be all that is needed — pass
 *        PartialOnlyTile::with_valid_reduce_dim_elements(n) for that.
 *        n must be in [1, tile reduce-axis dim].
 */
template <
    uint32_t dfb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    bool compute_uses_reduce_tile = false,
    typename PartialScalerT = NoPartial>
FORCE_INLINE void prepare_reduce_scaler(float scaler_f, PartialScalerT partial_scaler = PartialScalerT{});

/**
 * @brief Generate one or two reduce scaler tiles (per partial_scaler) with format and tile shape deduced from dfb_id
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
 * @tparam compute_uses_reduce_tile When true, forces row-0 fill (reduce LLK layout) even for
 *         SUM/AVG + REDUCE_ROW combinations that would normally use col-0 fill (matmul layout).
 *         Set to true when the compute kernel uses reduce_tile LLK directly instead of
 *         compute_kernel_lib::reduce (which auto-switches to matmul for REDUCE_ROW SUM/AVG).
 * @tparam PartialScalerT Partial-scaler descriptor type, deduced from `partial_scaler`.
 *         NoPartial (default) emits one full tile; PartialLastTile emits a full + partial
 *         tile pair for a non-tile-aligned reduce axis. In special cases where the compute
 *         side does not use the reduce helper, a single partial tile may be all that is
 *         needed — use PartialOnlyTile for that. REDUCE_SCALAR requires NoPartial.
 * @param partial_scaler Partial-scaler descriptor (default NoPartial{} = one full tile).
 *        Pass PartialLastTile::with_valid_reduce_dim_elements(n) for a non-tile-aligned
 *        reduce axis (full tile first, then partial; the scaler DFB must hold 2 tiles).
 *        In special cases where the compute side does not use the reduce helper, a single
 *        partial tile may be all that is needed — pass
 *        PartialOnlyTile::with_valid_reduce_dim_elements(n) for that.
 */
template <
    uint32_t dfb_id,
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t reduce_factor = SUM_AND_MAX_REDUCE_FACTOR,
    bool compute_uses_reduce_tile = false,
    typename PartialScalerT = NoPartial>
FORCE_INLINE void calculate_and_prepare_reduce_scaler(PartialScalerT partial_scaler = PartialScalerT{});

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
