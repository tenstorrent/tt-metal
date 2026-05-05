// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
/**
 * @file reduce_helpers_compute.hpp
 * @brief Single unified reduce function with automatic dispatch
 *
 * Provides ONE function that handles all reduce operations:
 * - Row reduction (REDUCE_ROW): Reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): Reduces H dimension, outputs Wt tiles per batch
 * - Scalar reduction (REDUCE_SCALAR): Reduces both H and W, outputs 1 tile per batch
 *
 * This library hides the complexity of:
 * - tile_regs_acquire/commit/wait/release DST register management
 * - reduce_init/reduce_uninit initialization
 * - Circular buffer manipulation (cb_wait_front, cb_pop_front, cb_reserve_back, cb_push_back)
 * - pack_tile for writing results to output CB
 * - Multiple input policies (see ReduceInputPolicy enum)
 *
 * DEST register capacity is automatically detected via dest_helpers.hpp.
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * IMPORTANT: The scaler CB must contain the scaling factor tile BEFORE calling reduce().
 *
 * Basic Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
 *
 *   compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
 *
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * See reduce() function documentation for advanced usage examples including:
 * - Different input policies (BulkWaitBulkPop, NoWaitNoPop, WaitUpfrontNoPop)
 * - Post-reduce operations (e.g., recip_tile for softmax)
 * - Accumulation for block-wise reduction
 */

namespace compute_kernel_lib {

// =============================================================================
// Reconfig Mode - control data format reconfiguration before reduce
// =============================================================================

/**
 * @brief Reconfiguration mode for data format unpacker and packer setup
 *
 * If the data format for input (unpacker) or output (packer) differs from what the
 * previous op configured, unpacker and/or packer must be reconfigured.
 *
 * Perf note: unnecessary reconfigurations cost cycles. If the caller tracks data-format
 * usage across consecutive ops it can pick a narrower mode. When that is impractical,
 * INPUT_AND_OUTPUT is the safe default (biggest perf hit, but always correct).
 *
 * - NONE: Skip all reconfiguration (reduce is first op, or input and output formats
 *         both match the previous op).
 * - INPUT: Reconfigure unpacker only (input CB format differs from previous op).
 * - OUTPUT: Reconfigure packer only (output CB format differs from previous op).
 * - INPUT_AND_OUTPUT: Reconfigure both (default, safest, largest perf impact).
 */
enum class ReduceDataFormatReconfigMode { NONE, INPUT, OUTPUT, INPUT_AND_OUTPUT };

// =============================================================================
// Input Policy - control how input tiles are synchronized and consumed
// =============================================================================

/**
 * @brief Input synchronization and consumption policy for reduce operations
 *
 * Controls when to wait for input tiles and whether to pop them after processing:
 *
 * - WaitAndPopPerTile: Wait/process/pop one tile at a time (streaming, safe for any CB size).
 *
 * - BulkWaitBulkPop: Wait for bulk, process all with indexed access, pop bulk.
 *   Bulk size depends on reduce dimension:
 *     REDUCE_SCALAR: Bulk = Ht×Wt tiles → 1 output per batch
 *     REDUCE_ROW:    Bulk = Wt tiles    → 1 output per row
 *     REDUCE_COL:    Bulk = Ht×chunk    → chunk outputs (chunk = DEST_AUTO_LIMIT)
 *
 * - WaitUpfrontNoPop: Wait for all tiles upfront, don't pop (persistent, for tile reuse).
 *   For REDUCE_COL tiles are indexed in standard row-major order (batch_offset + Ht*stride + Wt).
 *
 * - NoWaitNoPop: Caller manages wait/pop externally (preloaded, tiles already in CB).
 *   For REDUCE_COL tiles are accessed in row-major order, same as WaitUpfrontNoPop.
 */
enum class ReduceInputPolicy { WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop };

// =============================================================================
// Configuration Types
// =============================================================================

/**
 * @brief Input memory layout specification for PRELOADED/PERSISTENT reduce modes
 *
 * Specifies how input tiles are arranged in memory, particularly for non-contiguous layouts
 * where rows have padding (row_stride > logical width).
 */
struct ReduceInputMemoryLayout {
    uint32_t row_stride = 0;  // 0 = auto-detect from Wt (contiguous row-major)

    explicit constexpr ReduceInputMemoryLayout() = default;
    explicit constexpr ReduceInputMemoryLayout(uint32_t row) : row_stride(row) {}

    static constexpr ReduceInputMemoryLayout contiguous() { return ReduceInputMemoryLayout(); }
    static constexpr ReduceInputMemoryLayout with_row_stride(uint32_t s) { return ReduceInputMemoryLayout(s); }
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
struct ReduceInputBlockShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;

    static constexpr ReduceInputBlockShape of(uint32_t r, uint32_t c, uint32_t b = 1) { return {r, c, b}; }
    static constexpr ReduceInputBlockShape single() { return {1, 1, 1}; }
    static constexpr ReduceInputBlockShape row(uint32_t c, uint32_t b = 1) { return {1, c, b}; }
    static constexpr ReduceInputBlockShape col(uint32_t r, uint32_t b = 1) { return {r, 1, b}; }
};

/**
 * @brief Configuration for accumulation-style reductions
 *
 * Holds the static configuration for accumulation (CB and DST index).
 * Does not hold iteration state - that's provided via Accumulate wrapper.
 */
struct AccumulationConfig {
    // CB holding the running accumulator tile across reduce() iterations; see Accumulate below.
    uint32_t cb_accumulator = 0;
    uint32_t dst_index = 0;  // DST register for accumulation (default: 0)

    static constexpr AccumulationConfig with_cb(uint32_t cb, uint32_t dst = 0) { return {cb, dst}; }
};

/**
 * @brief Accumulation wrapper that carries config + iteration index
 *
 * This type enables type-based dispatch in reduce():
 * - When Accumulate is passed: accumulation code is compiled in
 * - When NoAccumulation (default): accumulation code is eliminated
 *
 * The iteration index determines reload behavior:
 * - iteration == 0: skip reload (first call, no accumulated value yet)
 * - iteration > 0: reload from accumulator CB before reducing
 *
 * Usage:
 *   const auto cfg = AccumulationConfig::with_cb(cb_accum);
 *   for (uint32_t i = 0; i < num_blocks; ++i) {
 *       reduce<SUM, REDUCE_ROW>(..., Accumulate(cfg, i));
 *   }
 *
 * Or with factory method:
 *   reduce<SUM, REDUCE_ROW>(..., Accumulate::at(cb_accum, iteration));
 */
struct Accumulate {
    AccumulationConfig config;
    uint32_t iteration = 0;

    explicit constexpr Accumulate(AccumulationConfig cfg, uint32_t iter = 0) : config(cfg), iteration(iter) {}
    explicit constexpr Accumulate(uint32_t cb, uint32_t iter = 0) : config{cb, 0}, iteration(iter) {}

    // Factory for concise call sites
    static constexpr Accumulate at(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }

    // Convenience: check if this is first iteration (skip reload)
    constexpr bool is_first() const { return iteration == 0; }
};

/**
 * @brief Tag type indicating no accumulation (zero overhead)
 *
 * When this type is passed to reduce(), all accumulation code is
 * eliminated at compile-time via `if constexpr`.
 */
struct NoAccumulation {};

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

/**
 * @brief Default no-op functor for post operation parameter
 *
 * When no custom post operation is needed, this empty functor is used.
 * It compiles away completely due to inlining.
 */
struct NoOp {
    ALWI void operator()(uint32_t = 0) const {}
};

// =============================================================================
// Main Reduce Function
// =============================================================================

/**
 * @brief Unified reduce function handling all reduction patterns
 *
 * This single function handles:
 * - Row reduction (REDUCE_ROW): Reduces W dimension, outputs Ht tiles per batch
 * - Column reduction (REDUCE_COL): Reduces H dimension, outputs Wt tiles per batch
 * - Scalar reduction (REDUCE_SCALAR): Reduces both H and W, outputs 1 tile per batch
 *
 * IMPORTANT - HARDWARE INITIALIZATION REQUIREMENT:
 * Before calling this function, you MUST initialize the compute kernel hardware by
 * calling compute_kernel_hw_startup() at the start of your kernel.
 *
 * IMPORTANT - SCALER CB REQUIREMENT:
 * The scaler CB (scaler_cb) must contain the scaling factor tile BEFORE calling
 * this function. The function will wait for it automatically.
 *
 * INPUT POLICIES: See ReduceInputPolicy enum for detailed mode descriptions.
 * - Use BulkWaitBulkPop for optimal performance when wait/pop are symmetric with ReduceInputBlockShape.
 * - Use NoWaitNoPop for asymmetric wait/pop (e.g., padding where you wait/pop more than ReduceInputBlockShape).
 * - Use WaitUpfrontNoPop for softmax patterns where tiles are reused in subsequent operations.
 *
 * POST-REDUCE OPERATIONS:
 * - post_reduce_op callback receives dst_idx parameter indicating which DEST register to operate on
 * - REDUCE_ROW: Called once per row with dst_idx=0 (single output in DST[0])
 * - REDUCE_COL: Called once per column in current chunk with dst_idx in [0, current_chunk)
 * - REDUCE_SCALAR: Called once per batch with dst_idx pointing at the single accumulated DST register
 *
 * @tparam reduce_type The type of reduce operation (SUM, AVG, MAX) - required explicit parameter
 * @tparam reduce_dim The dimension to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR) - required explicit parameter
 * @tparam input_policy Input handling policy (default: WaitAndPopPerTile - streaming mode)
 * @tparam reconfig_mode Data format reconfiguration mode (default: INPUT_AND_OUTPUT)
 *
 * @param input_cb Input circular buffer containing tiles to reduce
 * @param scaler_cb Circular buffer containing scaler tile
 * @param output_cb Output circular buffer for reduced tiles
 * @param input_block_shape Tile grid dimensions (rows x cols x batches)
 *              Use ReduceInputBlockShape::of(r, c, b), ::row(c), ::col(r), or ::single()
 * @param input_memory_layout Tile memory layout specification for NoWaitNoPop/WaitUpfrontNoPop policies (default:
 * contiguous) Use ReduceInputMemoryLayout::with_row_stride(stride) for custom row spacing. Only used when input_policy
 * is NoWaitNoPop or WaitUpfrontNoPop.
 * @param accumulate Accumulation configuration (default: NoAccumulation)
 * @param post_reduce_op Callback after each reduction (default: NoOp)
 *
 * @example
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * @example
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * @example
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * @example
 *   // Reduce type and dimension specified with explicit namespace
 *   compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::single());
 *
 * @example
 *   // NoWaitNoPop policy: caller manages wait/pop externally
 *   // Use cases: (1) custom stride between rows, (2) sharded CB mapped to tensor with data reuse
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
 *       cb_in, cb_scaler, cb_out, compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
 *       compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(input_stride));
 *
 * @example
 *   // WaitUpfrontNoPop policy: tiles persist for reuse (ideal for softmax pattern)
 *   // Library waits for tiles internally, but does NOT pop - tiles remain for subsequent ops
 *   compute_kernel_lib::reduce<MAX, REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
 *       cb_values, cb_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt));
 *   // cb_values tiles still available for sub_exp_block_bcast_cols_inplace()
 *
 * @example
 *   // BulkWaitBulkPop policy (bulk wait/pop - optimal for performance)
 *   // Library waits for all Wt tiles per row, processes them with indexed access, then pops all Wt tiles
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
 *       cb_in, cb_scaler, cb_out, compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * @example
 *   // Post-reduce operation: softmax pattern with recip_tile after SUM reduce
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
 *       cb_exps, cb_scaler, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt),
 *       compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
 *       NoAccumulation{},
 *       [](uint32_t dst_idx) {
 *           recip_tile_init();
 *           recip_tile(dst_idx);
 *       });
 *
 * @example
 *   // REDUCE_COL with post_reduce_op: apply recip_tile to each column result
 *   // dst_idx indicates which DEST register contains the column result (0 to current_chunk-1)
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt),
 *       {},
 *       NoAccumulation{},
 *       [](uint32_t dst_idx) {
 *           recip_tile_init();
 *           recip_tile(dst_idx);
 *       });
 */
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    ReduceInputPolicy input_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    typename AccumulateT = NoAccumulation,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t input_cb_id,
    uint32_t scaler_cb_id,
    uint32_t output_cb_id,
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{});

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl"
