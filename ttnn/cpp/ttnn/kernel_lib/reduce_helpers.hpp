// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
/**
 * @file reduce_helpers.hpp
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
 * - Multiple input modes: STREAMING (one-at-a-time), STREAMING_BATCHED (bulk), PRELOADED, PERSISTENT
 *
 * DEST register capacity is automatically detected via dest_helpers.hpp.
 *
 * IMPORTANT: Requires compute kernel hardware initialization.
 * Call compute_kernel_hw_startup() before using.
 *
 * IMPORTANT: The scaler CB must contain the scaling factor tile BEFORE calling reduce().
 *
 * Usage:
 *   #include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
 *
 *   compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
 *
 *   // STREAMING mode (default) - one-at-a-time for safety:
 *   // Library waits/pops each tile individually. No manual CB management needed!
 *
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::grid(Ht, Wt, NC));
 *
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::grid(Ht, Wt, NC));
 *
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::grid(Ht, Wt, NC));
 *
 *   // Reduce types and dimensions must now be specified explicitly as template parameters
 *   compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::grid(Ht, Wt, NC));
 */

namespace compute_kernel_lib {

// get_dest_limit() and DEST_AUTO_LIMIT are provided by dest_helpers.hpp

/**
 * @brief Input mode for reduce operations
 *
 * STREAMING: One-at-a-time mode - waits/pops each tile individually (default)
 *            Safe for numerical precision, compatible with any CB size.
 * STREAMING_BATCHED: Batched mode - waits for all tiles in row/batch, indexed access, pops all
 *                    Optimal performance when tiles are pre-loaded in CB.
 * PRELOADED: All tiles already present in CB, accessed via indexing (caller manages wait/pop)
 * PERSISTENT: Wait for all tiles upfront, indexed access, NO pop (tiles persist for reuse)
 */
enum class ReduceInputMode { STREAMING, STREAMING_BATCHED, PRELOADED, PERSISTENT };

/**
 * @brief Data format reconfiguration mode for reduce operations
 *
 * Controls whether the library automatically reconfigures the unpacker and packer
 * data formats before executing the reduce operation.
 *
 * NONE: No reconfig - use when reduce is first operation or formats already match
 * INPUT: Reconfig unpacker only (reconfig_data_format)
 * OUTPUT: Reconfig packer only (pack_reconfig_data_format)
 * BOTH: Reconfig both unpacker and packer (DEFAULT)
 */
enum class ReduceDataFormatReconfig { NONE = 0, INPUT = 1, OUTPUT = 2, BOTH = 3 };

/**
 * @brief Tile memory layout specification for PRELOADED/PERSISTENT reduce modes
 *
 * Specifies the stride pattern for accessing tiles in non-contiguous memory layouts.
 * Used only when input_mode is PRELOADED or PERSISTENT.
 */
struct TileLayout {
    uint32_t row_stride = 0;    // 0 = auto-detect from Wt (contiguous row-major)
    uint32_t batch_stride = 0;  // 0 = auto-detect from Ht * row_stride (reserved for future use)

    // Factory methods for common patterns
    static constexpr TileLayout contiguous() { return {}; }
    static constexpr TileLayout with_row_stride(uint32_t s) { return {s, 0}; }
    static constexpr TileLayout with_strides(uint32_t row, uint32_t batch) { return {row, batch}; }
};

/**
 * @brief Tile shape specification for reduce operations
 *
 * Specifies the grid dimensions (rows x cols x batches) for tile-based reductions.
 * Provides self-documenting factory methods for common patterns.
 */
struct TileShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t batches;

    // Full grid specification
    static constexpr TileShape grid(uint32_t r, uint32_t c, uint32_t b = 1) { return {r, c, b}; }

    // Single tile (1x1x1) - for scalar reductions on one tile
    static constexpr TileShape single() { return {1, 1, 1}; }

    // Single row of tiles (1 x cols x 1) - common for REDUCE_ROW
    static constexpr TileShape row(uint32_t c, uint32_t b = 1) { return {1, c, b}; }

    // Single column of tiles (rows x 1 x 1) - common for REDUCE_COL
    static constexpr TileShape col(uint32_t r, uint32_t b = 1) { return {r, 1, b}; }
};

/**
 * @brief Tag type indicating no accumulation (zero overhead)
 *
 * When this type is passed to reduce(), all accumulation code is
 * eliminated at compile-time via `if constexpr`.
 */
struct NoAccumulation {};

/**
 * @brief Configuration for accumulation-style reductions
 *
 * Holds the static configuration for accumulation (CB and DST index).
 * Does not hold iteration state - that's provided via Accumulate wrapper.
 */
struct AccumulationConfig {
    uint32_t cb_accumulator = 0;  // CB for accumulator
    uint32_t dst_index = 0;       // DST register for accumulation (default: 0)

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

    constexpr Accumulate(AccumulationConfig cfg, uint32_t iter = 0) : config(cfg), iteration(iter) {}
    constexpr Accumulate(uint32_t cb, uint32_t iter = 0) : config{cb, 0}, iteration(iter) {}

    // Factory for concise call sites
    static constexpr Accumulate at(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }

    // Convenience: check if this is first iteration (skip reload)
    constexpr bool is_first() const { return iteration == 0; }
};

/**
 * @brief Default no-op functor for post_reduce_op parameter
 *
 * When no custom post-reduce operation is needed, this empty functor is used.
 * It compiles away completely due to inlining.
 */
struct NoOp {
    ALWI void operator()(uint32_t = 0) const {}
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Short reduce init - only reconfigures unpacker and math, not packer
 *
 * This is needed after copy_tile_to_dst_init_short_with_dt which only reconfigures
 * the unpacker for copy operations. After the copy, we need to restore unpacker and
 * math config for reduce operations without touching the packer configuration.
 *
 * Equivalent to reduce_init but skips packer reconfiguration (llk_pack_reduce_mask_config).
 * Packer configuration from the initial reduce_init call remains valid.
 *
 * @tparam reduce_type The type of reduce operation (SUM, AVG, MAX)
 * @tparam reduce_dim The dimension to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR)
 * @tparam enforce_fp32_accumulation Enable accumulation in full FP32 precision
 * @param old_cbid The previous CB ID (accumulator CB) to reconfigure from
 * @param icb The input CB ID to reduce from
 * @param icb_scaler The scaler CB ID
 */
template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
ALWI void reduce_init_short_with_dt(uint32_t old_cbid, uint32_t icb, uint32_t icb_scaler) {
    // Reconfigure SRCA data format from old_cbid to icb (similar to copy_tile_to_dst_init_short_with_dt)
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, icb)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_cbid, icb)));

    // Reconfigure unpacker for reduce operation (SRCA and SRCB)
    UNPACK((llk_unpack_AB_reduce_init<reduce_dim, BroadcastType::NONE, enforce_fp32_accumulation>(icb, icb_scaler)));

    // Reconfigure math for reduce operation
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, enforce_fp32_accumulation>()));

    // Skip packer reconfiguration - it remains valid from initial reduce_init call
}

// Type trait to detect if AccumT is Accumulate (enables accumulation code)
template <typename T>
struct is_accumulate : std::false_type {};

template <>
struct is_accumulate<Accumulate> : std::true_type {};

template <typename T>
inline constexpr bool is_accumulate_v = is_accumulate<T>::value;

/**
 * @brief Helper to extract dst_index from accumulation type
 *
 * Returns the configured dst_index when AccumT is Accumulate,
 * or 0 (default) when AccumT is NoAccumulation.
 */
template <typename AccumT>
ALWI constexpr uint32_t get_dst_index(const AccumT& accum) {
    if constexpr (is_accumulate_v<AccumT>) {
        return accum.config.dst_index;
    } else {
        return 0;
    }
}

/**
 * @brief Helper function to reload accumulator tile into DST register
 *
 * When AccumT is Accumulate and iteration > 0:
 * 1. Loads accumulator tile from cb_accumulator to DST[dst_index]
 * 2. Re-initializes reduce operation (critical after copy_tile corrupts SRCA config)
 *
 * When AccumT is NoAccumulation: compiles to nothing (zero overhead)
 *
 * @tparam reduce_type The type of reduce operation (SUM, AVG, MAX)
 * @tparam reduce_dim The dimension to reduce (REDUCE_ROW, REDUCE_SCALAR, etc.)
 * @tparam AccumT Either Accumulate (enables accumulation) or NoAccumulation (zero overhead)
 * @tparam enforce_fp32_accumulation Whether to enforce FP32 accumulation
 */
template <PoolType reduce_type, ReduceDim reduce_dim, typename AccumT, bool enforce_fp32_accumulation>
ALWI void reload_accumulator_if_needed(uint32_t icb, uint32_t icb_scaler, const AccumT& accum) {
    if constexpr (is_accumulate_v<AccumT>) {
        if (!accum.is_first()) {  // Reload on all iterations except first
            constexpr uint32_t onetile = 1;
            cb_wait_front(accum.config.cb_accumulator, onetile);
            copy_tile_to_dst_init_short_with_dt(icb, accum.config.cb_accumulator);
            copy_tile(accum.config.cb_accumulator, 0, accum.config.dst_index);
            cb_pop_front(accum.config.cb_accumulator, onetile);

            // CRITICAL: Re-init reduce after copy_tile corrupts SRCA config
            // Use short version since packer config is still valid from initial reduce_init
            // Pass accumulator CB as old_cbid to reconfigure data format from accumulator to input CB
            reduce_init_short_with_dt<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                accum.config.cb_accumulator, icb, icb_scaler);
        }
    }
}

// =============================================================================
// Single Unified Reduce Function
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
 * The scaler CB (icb_scaler) must contain the scaling factor tile BEFORE calling
 * this function. The function will wait for it automatically when init=true.
 *
 * IMPORTANT - REDUCE_COL DATA LAYOUT:
 * - STREAMING mode: Tiles processed one-at-a-time in column-major chunks due to DEST limits.
 * - STREAMING_BATCHED mode: Tiles batched per chunk (Ht*chunk_size tiles), indexed access.
 * - PRELOADED/PERSISTENT mode: Tiles in standard row-major order (batch_offset + ht*stride + wt).
 * - Chunk size is auto-detected from DEST register capacity (DEST_AUTO_LIMIT).
 *
 * INPUT MODES: See ReduceInputMode enum for detailed mode descriptions.
 * - Use STREAMING_BATCHED for optimal performance when wait/pop are symmetric with TileShape.
 * - Use PRELOADED for asymmetric wait/pop (e.g., padding where you wait/pop more than TileShape).
 * - Use PERSISTENT for softmax patterns where tiles are reused in subsequent operations.
 *
 * POST-REDUCE OPERATIONS:
 * - post_reduce_op callback receives dst_idx parameter indicating which DEST register to operate on
 * - REDUCE_ROW: Called once per row with dst_idx=0 (single output in DST[0])
 * - REDUCE_COL: Called once per column in current chunk with dst_idx in [0, current_chunk)
 * - REDUCE_SCALAR: Not called (single reduction to DST[0])
 *
 * @tparam reduce_type The type of reduce operation (SUM, AVG, MAX) - required explicit parameter
 * @tparam reduce_dim The dimension to reduce (REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR) - required explicit parameter
 * @tparam input_mode Input handling mode (STREAMING, STREAMING_BATCHED, PRELOADED, PERSISTENT) - defaults to STREAMING
 * @tparam reconfig Data format reconfiguration mode (NONE, INPUT, OUTPUT, BOTH) - defaults to BOTH
 * @tparam init If true, calls reduce_init before processing (default: true)
 * @tparam uninit If true, calls reduce_uninit after processing (default: true)
 *
 * @note FP32 accumulation is auto-detected from ENABLE_FP32_DEST_ACC define via get_fp32_dest_acc_enabled()
 *
 * @param icb Input circular buffer containing tiles to reduce
 * @param icb_scaler Circular buffer containing scaler tile
 * @param ocb Output circular buffer for reduced tiles
 * @param shape Tile grid dimensions (rows x cols x batches)
 *              Use TileShape::grid(r, c, b), TileShape::row(c), TileShape::col(r), or TileShape::single()
 * @param layout Tile memory layout specification for PRELOADED/PERSISTENT modes (default: contiguous)
 *               Use TileLayout::with_row_stride(stride) for custom row spacing.
 *               Only used when input_mode is PRELOADED or PERSISTENT.
 *
 * @example
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::grid(Ht, Wt, NC));
 *
 * @example
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::grid(Ht, Wt, NC));
 *
 * @example
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::grid(Ht, Wt, NC));
 *
 * @example
 *   // Reduce type and dimension must be specified explicitly as template parameters
 *   compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::single());
 *
 * @example
 *   // PRELOADED mode: caller manages wait/pop, with custom stride between rows
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PRELOADED>(
 *       cb_in, cb_scaler, cb_out, compute_kernel_lib::TileShape::grid(Ht, Wt, NC),
 *       compute_kernel_lib::TileLayout::with_row_stride(input_stride));
 *
 * @example
 *   // PERSISTENT mode: tiles persist for reuse (ideal for softmax pattern)
 *   // Library waits for tiles internally, but does NOT pop - tiles remain for subsequent ops
 *   compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW,
 *                              compute_kernel_lib::ReduceInputMode::PERSISTENT>(
 *       cb_values, cb_scaler, cb_max, compute_kernel_lib::TileShape::grid(Ht, Wt));
 *   // cb_values tiles still available for sub_exp_block_bcast_cols_inplace()
 *
 * @example
 *   // STREAMING_BATCHED mode (optimal when tiles already in CB, symmetric wait/pop)
 *   // Library waits for all Wt tiles per row, processes them, then pops all Wt tiles
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputMode::STREAMING_BATCHED>(
 *       cb_in, cb_scaler, cb_out, compute_kernel_lib::TileShape::grid(Ht, Wt, NC));
 *
 * @example
 *   // Post-reduce operation: softmax pattern with recip_tile after SUM reduce
 *   // Set uninit=false since lambda calls reduce_uninit() before recip
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputMode::PRELOADED,
 *       true, false>(
 *       cb_exps, cb_scaler, cb_out, compute_kernel_lib::TileShape::row(Wt),
 *       compute_kernel_lib::TileLayout::contiguous(),
 *       [](uint32_t) {
 *           reduce_uninit();
 *           recip_tile_init();
 *           recip_tile(0);
 *       });
 *
 * @example
 *   // REDUCE_COL with post_reduce_op: apply recip_tile to each column result
 *   // dst_idx indicates which DEST register contains the column result (0 to current_chunk-1)
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::TileShape::grid(Ht, Wt),
 *       {},
 *       [](uint32_t dst_idx) {
 *           recip_tile_init();
 *           recip_tile(dst_idx);
 *       });
 */
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    ReduceInputMode input_mode = ReduceInputMode::STREAMING,
    ReduceDataFormatReconfig reconfig = ReduceDataFormatReconfig::BOTH,
    bool init = true,
    bool uninit = true,
    typename AccumT = NoAccumulation,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t ocb,
    TileShape shape,
    TileLayout layout = {},
    AccumT accum = {},
    PostReduceOp post_reduce_op = {}) {
    // Compile-time flag: true when Accumulate type is passed, false otherwise
    constexpr bool enable_accumulation = is_accumulate_v<AccumT>;
    // Extract shape components
    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t num_batches = shape.batches;

    // Apply reconfig based on mode
    if constexpr (reconfig == ReduceDataFormatReconfig::INPUT || reconfig == ReduceDataFormatReconfig::BOTH) {
        reconfig_data_format(icb, icb_scaler);
    }
    if constexpr (reconfig == ReduceDataFormatReconfig::OUTPUT || reconfig == ReduceDataFormatReconfig::BOTH) {
        pack_reconfig_data_format(ocb);
    }

    // Auto-detect FP32 dest accumulation mode from compile-time define
    constexpr bool enforce_fp32_accumulation = get_fp32_dest_acc_enabled();

    // Initialization
    if constexpr (init) {
        reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, ocb);
    }
    cb_wait_front(icb_scaler, 1);  // Wait for scaler tile

    constexpr uint32_t onetile = 1;

    // Pattern dispatch based on reduce_dim
    if constexpr (reduce_dim == ReduceDim::REDUCE_SCALAR) {
        // =================================================================
        // REDUCE_SCALAR: HW reduction - all tiles -> 1 output tile per batch
        // =================================================================
        const uint32_t stride = (layout.row_stride > 0) ? layout.row_stride : Wt;
        const uint32_t tiles_per_batch = Ht * stride;
        const uint32_t total_tiles = tiles_per_batch * num_batches;

        // PRELOADED: bulk reserve output upfront
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_reserve_back(ocb, num_batches);
        }

        // PERSISTENT: wait for all tiles upfront
        if constexpr (input_mode == ReduceInputMode::PERSISTENT) {
            cb_wait_front(icb, total_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            // STREAMING_BATCHED: wait for all tiles upfront
            if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                cb_wait_front(icb, tiles_per_batch);
            }

            tile_regs_acquire();

            // Reload accumulator if needed (zero overhead when AccumT is NoAccumulation)
            reload_accumulator_if_needed<reduce_type, reduce_dim, AccumT, enforce_fp32_accumulation>(
                icb, icb_scaler, accum);

            const uint32_t dst_idx = get_dst_index(accum);
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (input_mode == ReduceInputMode::STREAMING) {
                        // One-at-a-time: wait/pop per tile
                        cb_wait_front(icb, onetile);
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, 0, 0, dst_idx);
                        cb_pop_front(icb, onetile);
                    } else if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                        // Batched: use indexed access
                        uint32_t tile_idx = ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            icb, icb_scaler, tile_idx, 0, dst_idx);
                    } else {  // PRELOADED or PERSISTENT: indexed access
                        uint32_t tile_idx = batch_offset + ht * stride + wt;
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            icb, icb_scaler, tile_idx, 0, dst_idx);
                    }
                }
            }
            // STREAMING/STREAMING_BATCHED/PERSISTENT: reserve per-batch
            if constexpr (
                input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::STREAMING_BATCHED ||
                input_mode == ReduceInputMode::PERSISTENT) {
                cb_reserve_back(ocb, onetile);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(get_dst_index(accum), ocb);
            tile_regs_release();
            if constexpr (
                input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::STREAMING_BATCHED ||
                input_mode == ReduceInputMode::PERSISTENT) {
                cb_push_back(ocb, onetile);
            }

            // STREAMING_BATCHED: pop all tiles after processing
            if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                cb_pop_front(icb, tiles_per_batch);
            }

            // PRELOADED or PERSISTENT: update batch offset
            if constexpr (input_mode == ReduceInputMode::PRELOADED || input_mode == ReduceInputMode::PERSISTENT) {
                batch_offset += tiles_per_batch;
            }
        }

        // PRELOADED: bulk push output at end
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_push_back(ocb, num_batches);
        }
    } else if constexpr (reduce_dim == ReduceDim::REDUCE_ROW) {
        // =================================================================
        // REDUCE_ROW: W reduction - each row -> 1 output tile (Ht outputs per batch)
        // =================================================================
        const uint32_t stride = (layout.row_stride > 0) ? layout.row_stride : Wt;
        const uint32_t total_outputs = Ht * num_batches;
        const uint32_t total_tiles = Ht * stride * num_batches;

        // PRELOADED: bulk reserve output upfront
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_reserve_back(ocb, total_outputs);
        }

        // PERSISTENT: wait for all tiles upfront
        if constexpr (input_mode == ReduceInputMode::PERSISTENT) {
            cb_wait_front(icb, total_tiles);
        }

        uint32_t index_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                // STREAMING_BATCHED: wait for entire row upfront
                if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                    cb_wait_front(icb, Wt);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumT, enforce_fp32_accumulation>(
                    icb, icb_scaler, accum);

                const uint32_t dst_idx = get_dst_index(accum);
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    if constexpr (input_mode == ReduceInputMode::STREAMING) {
                        // One-at-a-time: wait/pop per tile
                        cb_wait_front(icb, onetile);
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(icb, icb_scaler, 0, 0, dst_idx);
                        cb_pop_front(icb, onetile);
                    } else if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                        // Batched: use indexed access
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            icb, icb_scaler, wt, 0, dst_idx);
                    } else {  // PRELOADED or PERSISTENT: indexed access
                        reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                            icb, icb_scaler, wt + index_offset, 0, dst_idx);
                    }
                }

                // Call post-reduce operation (e.g., recip_tile for softmax)
                // User's lambda can include reduce_uninit() if needed before custom ops
                post_reduce_op(dst_idx);

                // STREAMING/STREAMING_BATCHED/PERSISTENT: reserve per-row to avoid deadlock
                if constexpr (
                    input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::STREAMING_BATCHED ||
                    input_mode == ReduceInputMode::PERSISTENT) {
                    cb_reserve_back(ocb, onetile);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_idx, ocb);
                tile_regs_release();
                if constexpr (
                    input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::STREAMING_BATCHED ||
                    input_mode == ReduceInputMode::PERSISTENT) {
                    cb_push_back(ocb, onetile);
                }

                // STREAMING_BATCHED: pop all tiles after processing
                if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                    cb_pop_front(icb, Wt);
                }

                // PRELOADED or PERSISTENT: update index offset
                if constexpr (input_mode == ReduceInputMode::PRELOADED || input_mode == ReduceInputMode::PERSISTENT) {
                    index_offset += stride;
                }
            }
        }

        // PRELOADED: bulk push output at end
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_push_back(ocb, total_outputs);
        }
    } else {
        // =================================================================
        // REDUCE_COL: H reduction - each column -> 1 output tile (Wt outputs per batch)
        // Need chunking due to DEST register limits
        // STREAMING: Tiles arrive in N C W_skip H W_chunk order (chunked by chunk_size)
        // PRELOADED: Tiles in row-major order, indexed as batch_offset + ht*stride + wt
        // =================================================================

        // Auto-detect chunk size from DEST register capacity
        // Both reader (dataflow) and compute kernels compute this identically via DEST_AUTO_LIMIT
        constexpr uint32_t chunk_size = DEST_AUTO_LIMIT;
        const uint32_t stride = (layout.row_stride > 0) ? layout.row_stride : Wt;
        const uint32_t tiles_per_batch = Ht * stride;
        const uint32_t total_outputs = Wt * num_batches;
        const uint32_t total_tiles = tiles_per_batch * num_batches;

        // PRELOADED: bulk reserve output upfront
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_reserve_back(ocb, total_outputs);
        }

        // PERSISTENT: wait for all tiles upfront
        if constexpr (input_mode == ReduceInputMode::PERSISTENT) {
            cb_wait_front(icb, total_tiles);
        }

        uint32_t batch_offset = 0;
        for (uint32_t nc = 0; nc < num_batches; ++nc) {
            for (uint32_t wt = 0; wt < Wt; wt += chunk_size) {
                uint32_t chunk_end = (wt + chunk_size < Wt) ? (wt + chunk_size) : Wt;
                uint32_t current_chunk = chunk_end - wt;
                uint32_t tiles_in_chunk = Ht * current_chunk;

                // STREAMING_BATCHED: wait for entire chunk upfront
                if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                    cb_wait_front(icb, tiles_in_chunk);
                }

                tile_regs_acquire();

                // Reload accumulator if needed (zero overhead when AccumT is NoAccumulation)
                reload_accumulator_if_needed<reduce_type, reduce_dim, AccumT, enforce_fp32_accumulation>(
                    icb, icb_scaler, accum);

                for (uint32_t ht = 0; ht < Ht; ++ht) {
                    // Base dst_index: from accumulation config or 0 for multi-column output
                    uint32_t dst_idx = get_dst_index(accum);
                    for (uint32_t i = wt; i < chunk_end; ++i) {
                        if constexpr (input_mode == ReduceInputMode::STREAMING) {
                            // One-at-a-time: wait/pop per tile
                            cb_wait_front(icb, onetile);
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                icb, icb_scaler, 0, 0, dst_idx);
                            cb_pop_front(icb, onetile);
                        } else if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                            // Batched: use indexed access
                            uint32_t tile_idx = ht * current_chunk + (i - wt);
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                icb, icb_scaler, tile_idx, 0, dst_idx);
                        } else {  // PRELOADED or PERSISTENT: indexed access
                            uint32_t tile_idx = batch_offset + ht * stride + i;
                            reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(
                                icb, icb_scaler, tile_idx, 0, dst_idx);
                        }
                        ++dst_idx;
                    }
                }

                // Post-reduce operation for each output tile in chunk
                const uint32_t base_dst = get_dst_index(accum);
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    post_reduce_op(base_dst + i);
                }

                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < current_chunk; ++i) {
                    // STREAMING/STREAMING_BATCHED/PERSISTENT: reserve/push per output tile
                    if constexpr (
                        input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::STREAMING_BATCHED ||
                        input_mode == ReduceInputMode::PERSISTENT) {
                        cb_reserve_back(ocb, onetile);
                    }
                    pack_tile(base_dst + i, ocb);
                    if constexpr (
                        input_mode == ReduceInputMode::STREAMING || input_mode == ReduceInputMode::STREAMING_BATCHED ||
                        input_mode == ReduceInputMode::PERSISTENT) {
                        cb_push_back(ocb, onetile);
                    }
                }
                tile_regs_release();

                // STREAMING_BATCHED: pop all tiles after processing
                if constexpr (input_mode == ReduceInputMode::STREAMING_BATCHED) {
                    cb_pop_front(icb, tiles_in_chunk);
                }
            }
            // Update batch_offset for indexed modes (PRELOADED and PERSISTENT)
            if constexpr (input_mode == ReduceInputMode::PRELOADED || input_mode == ReduceInputMode::PERSISTENT) {
                batch_offset += tiles_per_batch;
            }
        }

        // PRELOADED: bulk push output at end
        if constexpr (input_mode == ReduceInputMode::PRELOADED) {
            cb_push_back(ocb, total_outputs);
        }
    }

    // Cleanup
    if constexpr (uninit) {
        reduce_uninit<enforce_fp32_accumulation>();
    }
}

}  // namespace compute_kernel_lib
