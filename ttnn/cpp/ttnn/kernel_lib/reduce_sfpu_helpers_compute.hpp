// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

/**
 * @file reduce_sfpu_helpers_compute.hpp
 * @brief SFPU path for tile reduction (Int32 + Float32) when the FPU GMPOOL reduce path is
 *        invalid (Int32) or imprecise (Float32: GMPOOL bf16-truncates SrcA/SrcB).
 *
 * Provides ONE function, `reduce_sfpu`, as the SFPU counterpart to `compute_kernel_lib::reduce`
 * in reduce_helpers_compute.hpp (same streaming WaitAndPopPerTile-style flow for one axis):
 * - Row reduction (REDUCE_ROW): reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): reduces H dimension, outputs Wt tiles per batch
 *
 * There is no REDUCE_SCALAR here; full HxW reduction uses two passes on the host, like the FPU path.
 *
 * This library hides the complexity of:
 * - tile_regs_acquire/commit/wait/release DST register management
 * - init_sfpu / copy_tile_to_dst_init_short (SFPU path; no GMPOOL reduce_tile)
 * - Circular buffer manipulation (cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back)
 * - pack_tile for writing results to output CB
 * - Packer reduce mask setup (sfpu_reduce does not configure it; this helper does)
 *
 * Within-tile reduction uses SFPU `sfpu_reduce`; cross-tile folds along the reduce axis use
 * `binary_max[_int32]_tile` only (selected by `format`). MIN is not folded here; host launches
 * `reduce_sfpu_{h,w}_neg.cpp` which implements MIN as -MAX(-x) and reuses the same MAX fold.
 *
 * IMPORTANT: Do not call `compute_kernel_hw_startup()` before `reduce_sfpu`. This helper runs
 * on the SFPU path and calls `init_sfpu` / `copy_tile_to_dst_init_short` itself.
 *
 * IMPORTANT: The scaler CB must contain the scaling factor tile BEFORE calling reduce_sfpu().
 * `sfpu_reduce` does not consume it; the helper waits and pops it so dataflow matches reduce().
 *
 * Basic Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.hpp"
 *
 *   // Row reduction (W) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce_sfpu<ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW, DataFormat::Float32>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
 *       post_mul_scaler_bits);
 *
 *   // Column reduction (H) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce_sfpu<ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_COL, DataFormat::Int32>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
 *       post_mul_scaler_bits);
 *
 * See reduce_sfpu() below for template parameters and post_mul_scaler_bits (used when
 * REDUCE_POST_MUL is defined by the host).
 */

namespace compute_kernel_lib {

/**
 * @brief SFPU reduce along one tile axis (templates match host REDUCE_* defines).
 *
 * @tparam pool_type   PoolType::MAX (MIN is dispatched via reduce_sfpu_{h,w}_neg.cpp).
 * @tparam reduce_dim  ReduceDim::REDUCE_ROW (W) or ReduceDim::REDUCE_COL (H).
 * @tparam format      DataFormat::Int32 or DataFormat::Float32.
 *
 * @param input_cb_id        Tiles to reduce (streaming order matches `reduce()` for same dim).
 * @param scaler_cb_id       Scaler tile CB (waited/popped; not passed into sfpu_reduce).
 * @param output_cb_id       Reduced output tiles.
 * @param input_block_shape  Tile grid (Ht, Wt, NC).
 * @param post_mul_scaler_bits  Packed fp32 user scalar; used only when `REDUCE_POST_MUL` is set.
 */
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat format>
ALWI void reduce_sfpu(
    uint32_t input_cb_id,
    uint32_t scaler_cb_id,
    uint32_t output_cb_id,
    ReduceInputBlockShape input_block_shape,
    uint32_t post_mul_scaler_bits);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.inl"
