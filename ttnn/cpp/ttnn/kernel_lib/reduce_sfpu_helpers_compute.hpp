// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

/**
 * @file reduce_sfpu_helpers_compute.hpp
 * @brief SFPU-based reduce parametrised by op x dim x format.
 *
 * This is the SFPU sibling of `compute_kernel_lib::reduce` (in
 * reduce_helpers_compute.hpp), which targets the FPU's GMPOOL primitive via
 * `reduce_tile`. GMPOOL silently produces zeros for INT32 inputs (issue
 * #26726, ttnn.sum repro in #26724), so this helper instead routes through
 * the SFPU primitives
 *  - `sfpu_reduce<OP, FMT, DIM>(...)` for the within-tile reduction
 *    (32 lanes -> 1 lane along the chosen axis), and
 *  - `binary_max_int32_tile` / `binary_min_int32_tile` / `add_int_tile` for
 *    the cross-tile accumulator fold (Wt or Ht tiles -> 1 tile of
 *    element-wise max/min/sum).
 *
 * Issue #43736 phases implemented here (Int32 only):
 *   - Phase 1: pool_type in {MAX, MIN}, reduce_dim in {REDUCE_ROW, REDUCE_COL}.
 *   - SUM extension (#26724): pool_type = SUM, reduce_dim in {REDUCE_ROW,
 *     REDUCE_COL}.  Cross-tile fold uses `add_int_tile<Int32>` (full-tile
 *     add) since `sum(sum(a, b), c) == sum(a, b, c)`; within-tile reduction
 *     still uses `sfpu_reduce<SUM, Int32, DIM>`.
 *
 * LLK gap (statically rejected): `sfpu_reduce<MIN, *, REDUCE_ROW>` is not in
 * the LLK, so MIN with REDUCE_ROW is rejected at compile time and the host
 * dispatch must gate it before reaching this helper.  The host can still
 * compute MIN via the negate-trick (`-MAX(-x)`) by passing `negate=true`
 * and `pool_type=MAX` -- the negate path mirrors the FPU's reduce_w_neg /
 * reduce_h_neg kernels.  The negate trick is meaningful only for MAX/MIN;
 * SUM with `negate=true` is statically rejected.
 *
 * Out of scope (reserved for follow-up phases):
 *   - AVG (will be lowered to SUM with a post-multiply by 1/N at the host).
 *   - UInt32, UInt16, Float32, Float16_b formats.
 *
 * Pre-conditions identical to compute_kernel_lib::reduce:
 *   - `compute_kernel_hw_startup` and `init_sfpu` must NOT be called by the
 *     caller -- this helper does its own SFPU-specific init.
 *   - The scaler CB must contain a tile pushed by the reader (we don't use
 *     it because sfpu_reduce takes no scaler, but we wait/pop it so the
 *     dataflow kernels can be shared with the FPU path unchanged).
 *
 * Tile arrival order:
 *   - REDUCE_ROW (W reduce): row-major (NC * Ht rows, each of Wt tiles).
 *   - REDUCE_COL (H reduce): the H reader is run with row_chunk=1 so tiles
 *     arrive one tile-column at a time (Ht tiles per output, contiguous
 *     along H).  The host program factory is responsible for setting up
 *     the reader that way.
 *
 * Usage (mirrors the FPU helper):
 *   #include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.hpp"
 *
 *   compute_kernel_lib::reduce_sfpu<
 *       PoolType::MAX,
 *       ReduceDim::REDUCE_ROW,
 *       DataFormat::Int32,
 *       false>(  // negate -- pass true to lower MIN to -MAX(-x)
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 */

namespace compute_kernel_lib {

/**
 * @brief SFPU-based reduce parameterised by op x dim x format.
 *
 * @tparam pool_type   PoolType::MAX, PoolType::MIN, or PoolType::SUM.  AVG is
 *                     reserved for a later phase (lowered to SUM at the host)
 *                     and statically rejected here.
 * @tparam reduce_dim  ReduceDim::REDUCE_ROW (W axis) or REDUCE_COL (H axis).
 *                     Note: pool_type MIN with REDUCE_ROW is not in the LLK
 *                     and is statically rejected.
 * @tparam format      Currently only DataFormat::Int32 is implemented.
 * @tparam negate      When true the helper computes `-pool_type(-x)` -- input
 *                     tiles are negated in DST before the cross-tile fold and
 *                     the within-tile sfpu_reduce, and the result is negated
 *                     before pack.  The host uses this to lower MIN to MAX
 *                     (mirrors the FPU reduce_w_neg / reduce_h_neg path).
 *                     Only valid for MAX/MIN; SUM with negate=true is
 *                     statically rejected (sum negate-trick has no use case).
 *
 * @param input_cb_id    Input CB containing the tiles to reduce.
 * @param scaler_cb_id   Scaler CB pushed by the reader (drained but unused).
 * @param output_cb_id   Output CB that receives the reduced tiles.
 * @param input_block_shape Input tile-grid shape (Ht, Wt, NC).  Output count
 *                       is Ht*NC for REDUCE_ROW and Wt*NC for REDUCE_COL.
 */
// Note on namespacing: `PoolType` and `ReduceDim` live in `namespace ckernel`, but
// `DataFormat` is at global scope (defined in tensix_types.h with no enclosing
// namespace).  We use the unqualified spellings here since the function template
// parameters resolve via standard unqualified lookup.
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat format, bool negate = false>
ALWI void reduce_sfpu(
    uint32_t input_cb_id, uint32_t scaler_cb_id, uint32_t output_cb_id, ReduceInputBlockShape input_block_shape);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.inl"
