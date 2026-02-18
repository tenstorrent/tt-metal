// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace dataflow_kernel_lib {

using ckernel::PoolType;
using ckernel::ReduceDim;

/**
 * @brief Generate a reduce scaler tile (legacy, bfloat16 only)
 *
 * Creates a tile in the specified circular buffer with the scaler value
 * placed in row 0 of each face. The scaler is typically 1.0 for SUM/MAX
 * reductions, and 1/N for AVG reductions.
 *
 * @tparam half_tile If true, only fill faces 0-1 (half tile mode)
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler Packed bf16 value (bf16 << 16 | bf16)
 */
template <bool half_tile = false>
FORCE_INLINE void generate_reduce_scaler_legacy(const uint32_t cb_id, const uint32_t scaler);

// =============================================================================
// Reduce scaler helpers API
//
// generate_reduce_scaler: Automatically computes the correct scaler based on
//   pool type, reduce dimension, and reduce factor (e.g. 1/N for AVG, 1.0 for
//   SUM/MAX). Use this when performing standard reduce operations.
//
// fill_cb_with_scaler: Fills a CB tile with a caller-provided float scaler.
//   Use this when you need a custom scaler value that doesn't follow the
//   standard reduce conventions. Accepts a runtime float value.
// =============================================================================

/**
 * @brief Fill a circular buffer tile with a scaler value
 *
 * Converts the float scaler to the appropriate bit representation based on
 * the circular buffer's data format, then fills row 0 of each face.
 * Data format and tile shape (half/full) are deduced from the circular buffer.
 *
 * @tparam cb_id Circular buffer ID to write the tile to (must be constexpr)
 * @param scaler_f Float scaler value to fill the tile with
 */
template <uint32_t cb_id>
FORCE_INLINE void fill_cb_with_scaler(float scaler_f);

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
 * @tparam reduce_factor Number of elements being reduced (N). Must be non-zero for AVG.
 * @param input_scaler Multiplicative scaler applied to the computed value (default 1.0f)
 */
template <uint32_t cb_id, PoolType pool_type, ReduceDim reduce_dim, uint32_t reduce_factor = 1>
FORCE_INLINE void generate_reduce_scaler(float input_scaler = 1.0f);

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
