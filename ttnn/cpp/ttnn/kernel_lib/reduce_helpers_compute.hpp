// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "tt-metalium/circular_buffer_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helper_policies.hpp"
/**
 * @file reduce_helpers_compute.hpp
 * @brief Single unified reduce function with automatic dispatch
 *
 * Provides ONE function that handles all reduce operations:
 * - Row reduction (REDUCE_ROW): Reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): Reduces H dimension, outputs Wt tiles per batch
 * - Scalar reduction (REDUCE_SCALAR): Reduces both H and W, outputs 1 tile per batch
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * IMPORTANT: The scaler CB must contain the scaling factor tile BEFORE calling reduce().
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
 *
 *   compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
 *
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::InputBlockShape::of(Ht, Wt, NC));
 *
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::InputBlockShape::of(Ht, Wt, NC));
 *
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::InputBlockShape::of(Ht, Wt, NC));
 */

namespace compute_kernel_lib {

// =============================================================================
// Configuration Types
// =============================================================================

/**
 * @brief Input memory layout specification for PRELOADED/PERSISTENT reduce modes
 *
 * Specifies how input tiles are arranged in memory, particularly for non-contiguous layouts
 * where rows have padding (row_stride > logical width).
 */
struct InputMemoryLayout {
    uint32_t row_stride = 0;  // 0 = auto-detect from Wt (contiguous row-major)

    explicit constexpr InputMemoryLayout() = default;
    explicit constexpr InputMemoryLayout(uint32_t row) : row_stride(row) {}

    static constexpr InputMemoryLayout contiguous() { return InputMemoryLayout(); }
    static constexpr InputMemoryLayout with_row_stride(uint32_t s) { return InputMemoryLayout(s); }
};

/**
 * @brief Input block shape specification for reduce operations (rows x cols x batches)
 *
 * Specifies the dimensions of the input tile block to be reduced.
 * The output size depends on the reduction dimension:
 * - REDUCE_ROW: output has (rows × batches) tiles
 * - REDUCE_COL: output has (cols × batches) tiles
 * - REDUCE_SCALAR: output has (batches) tiles
 */
struct InputBlockShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;

    static constexpr InputBlockShape of(uint32_t r, uint32_t c, uint32_t b = 1) { return {r, c, b}; }
    static constexpr InputBlockShape single() { return {1, 1, 1}; }
    static constexpr InputBlockShape row(uint32_t c, uint32_t b = 1) { return {1, c, b}; }
    static constexpr InputBlockShape col(uint32_t r, uint32_t b = 1) { return {r, 1, b}; }
};

// NoAccumulation is defined in common_types.hpp

/**
 * @brief Configuration for accumulation-style reductions
 */
struct AccumulationConfig {
    uint32_t cb_accumulator = 0;
    uint32_t dst_index = 0;

    static constexpr AccumulationConfig with_cb(uint32_t cb, uint32_t dst = 0) { return {cb, dst}; }
};

/**
 * @brief Accumulation wrapper that carries config + iteration index
 */
struct Accumulate {
    AccumulationConfig config;
    uint32_t iteration = 0;

    constexpr Accumulate(AccumulationConfig cfg, uint32_t iter = 0) : config(cfg), iteration(iter) {}
    constexpr Accumulate(uint32_t cb, uint32_t iter = 0) : config{cb, 0}, iteration(iter) {}

    static constexpr Accumulate at(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }

    constexpr bool is_first() const { return iteration == 0; }
};

// NoOp is defined in common_types.hpp

// =============================================================================
// Type Traits
// =============================================================================

template <typename T>
struct is_accumulate : std::false_type {};

template <>
struct is_accumulate<Accumulate> : std::true_type {};

template <typename T>
inline constexpr bool is_accumulate_v = is_accumulate<T>::value;

/**
 * @brief Type trait to detect valid accumulation types (NoAccumulation or Accumulate)
 */
template <typename T>
struct is_accumulation_type : std::false_type {};

template <>
struct is_accumulation_type<NoAccumulation> : std::true_type {};

template <>
struct is_accumulation_type<Accumulate> : std::true_type {};

template <typename T>
inline constexpr bool is_accumulation_type_v = is_accumulation_type<T>::value;

/**
 * @brief Type trait to detect valid post-reduce operation (callable with uint32_t)
 */
template <typename T, typename = void>
struct is_post_reduce_op : std::false_type {};

template <typename T>
struct is_post_reduce_op<T, std::void_t<decltype(std::declval<T>()(std::declval<uint32_t>()))>> : std::true_type {};

template <typename T>
inline constexpr bool is_post_reduce_op_v = is_post_reduce_op<T>::value;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Short reduce init - reconfigures unpacker and math after copy_tile
 */
template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
ALWI void reduce_init_short_with_dt(uint32_t old_cbid, uint32_t input_cb, uint32_t scaler_cb);

/**
 * @brief Helper to extract dst_index from accumulation type
 */
template <typename AccumulateT>
ALWI constexpr uint32_t get_dst_index(const AccumulateT& accumulate);

/**
 * @brief Helper to reload accumulator tile into DST register
 */
template <PoolType reduce_type, ReduceDim reduce_dim, typename AccumulateT, bool enforce_fp32_accumulation>
ALWI void reload_accumulator_if_needed(uint32_t input_cb, uint32_t scaler_cb, const AccumulateT& accumulate);

// =============================================================================
// Main Reduce Function
// =============================================================================

/**
 * @brief Unified reduce function handling all reduction patterns
 *
 * @tparam reduce_type The type of reduce operation (SUM, AVG, MAX)
 * @tparam reduce_dim The dimension to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 * @tparam InputPolicy Input handling policy (default: StreamingPolicy)
 * @tparam ReconfigPolicy Data format reconfiguration policy (default: ReconfigBothPolicy)
 *
 * @param input_cb Input circular buffer containing tiles to reduce
 * @param scaler_cb Circular buffer containing scaler tile
 * @param output_cb Output circular buffer for reduced tiles
 * @param input_block_shape Input block dimensions in tiles (rows x cols x batches)
 * @param input_memory_layout Input memory layout specification (default: contiguous)
 * @param accumulate Accumulation configuration (default: NoAccumulation)
 * @param post_reduce_op Callback after each reduction (default: NoOp)
 */
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    typename InputPolicy = reduce_policies::StreamingPolicy,
    typename ReconfigPolicy = reduce_policies::ReconfigBothPolicy,
    typename AccumulateT = NoAccumulation,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t input_cb,
    uint32_t scaler_cb,
    uint32_t output_cb,
    InputBlockShape input_block_shape,
    InputMemoryLayout input_memory_layout = InputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{});

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl"
