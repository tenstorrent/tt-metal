// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
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
 * - DataflowBuffer manipulation (wait_front, pop_front, reserve_back, push_back)
 * - pack_tile for writing results to output DFB
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
 *   compute_kernel_hw_startup(dfb_in, dfb_scaler, dfb_out);
 *
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, dfb_in, dfb_scaler, dfb_out>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL, dfb_in, dfb_scaler, dfb_out>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR, dfb_in, dfb_scaler, dfb_out>(
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
 * - WaitAndPopPerTile: Stream tiles through the input CB. ReduceTile processes one tile at a
 *   time. AccumulateViaAdd processes pairs and requires at least two input pages for an even
 *   reduce-tile count, or three pages for an odd count greater than one.
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
// Algorithm - which datapath implements the reduce
// =============================================================================

/**
 * @brief Which datapath implements the reduce.
 *
 * Auto is the default. It selects AccumulateViaAdd when all of the following hold:
 *   - standard (unit-scaler) floating-point SUM in Fast fp32 mode,
 *   - an input policy supported by AccumulateViaAdd (all combinations except
 *     WaitAndPopPerTile with REDUCE_COL),
 *   - a contiguous, tile-aligned input block,
 *   - either NoAccumulation or cross-call Accumulate.
 * Everything else stays on ReduceTile. In particular, callers using prepare_reduce_scaler with a
 * non-unit SUM scaler must select ReduceTile explicitly if they otherwise match the Auto fast path.
 *
 * AccumulateViaAdd is also an explicitly selectable float SUM/AVG datapath: it adds the
 * reduce-dimension tiles into one DST register, then performs the within-tile collapse with
 * sfpu_reduce.
 */
enum class ReduceAlgorithm { Auto, ReduceTile, AccumulateViaAdd };

/**
 * @brief How AccumulateViaAdd folds a running cross-call accumulator into a later chunk.
 *
 * FoldViaAdd is the fastest option, but reads the accumulator through SrcA/SrcB and therefore
 * requires an accumulator CB using UnpackToDestMode::Default. The CopySeed modes reload with
 * copy_tile and are safe for an UnpackToDestFp32 accumulator as well.
 *
 * CopySeedPairs is the safe default: it handles an odd tile with one DEST-reuse add and then
 * consumes the remaining tiles in pairs. CopySeedUniform uses one DEST-reuse add per tile.
 * CopySeedSfpuAdd sums the new chunk separately and combines it with the accumulator in the SFPU
 * (Wormhole/Blackhole only). CopySeedZeroPair pairs an odd tile with a zero tile supplied in the
 * scaler CB; it is limited to aligned reductions because partial reductions use that CB for a mask.
 */
enum class AccumulateReloadMode { FoldViaAdd, CopySeedPairs, CopySeedUniform, CopySeedSfpuAdd, CopySeedZeroPair };

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
 * @brief Partial-scaler descriptor for non-tile-aligned reduce dimensions
 *
 * When the reduce dimension is not a multiple of TILE_DIM, the reader emits
 * either a full scaler followed by a partial scaler, or just a partial scaler
 * when the input has only one tile along the reduce dimension. The compute
 * kernel must use the partial scaler for the *last* tile along the reduce
 * dimension, so the padding lanes multiply by zero and contribute nothing.
 *
 * This struct describes the scaler-CB layout. The default (`none()`) keeps the
 * legacy behavior of using one full scaler tile. Use `last_tile()` when the CB
 * contains [full, partial], or `only_tile()` when it contains one partial scaler
 * and the input has exactly one tile along the reduce dimension.
 *
 * Pair last_tile() with dataflow_kernel_lib::prepare_partial_reduce_scalers
 * (or calculate_and_prepare_partial_reduce_scalers) on the reader side. Pair
 * only_tile() with prepare_reduce_scaler using the valid reduce-axis count.
 *
 * REDUCE_SCALAR does not support partial scalers — it applies the scaler
 * twice (row then col), which a single partial tile cannot encode. The
 * runtime asserts that REDUCE_SCALAR callers pass none().
 *
 * The Int32 SFPU reduce path (see is_sfpu_reduce_path) folds tiles without
 * ever reading the scaler CB, so it cannot honor a partial scaler either; the
 * runtime asserts that those callers pass none() as well.
 *
 * IMPORTANT - this describes the last tile along the reduce dimension *of this
 * reduce() call*. If the caller collapses several tiles into one by element-wise
 * accumulation (add_tiles etc.) BEFORE reducing, a partial scaler is wrong: lane
 * j >= partial_positions of the accumulated tile holds padding from the ragged
 * tile but also VALID data from every earlier tile, and the partial scaler zeroes
 * all of it. Such callers must keep masking the ragged tile before accumulating
 * and reduce with a full scaler. (This is distinct from the Accumulate note
 * below, which is about accumulation *between* reduce() calls.)
 *
 * Usage:
 *   constexpr auto partial = has_partial
 *       ? ReducePartialScaler::last_tile()
 *       : ReducePartialScaler::none();
 *   reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, shape, ..., partial);
 */
enum class ReducePartialScalerMode : uint8_t {
    None,
    LastTile,
    OnlyTile,
};

struct ReducePartialScaler {
    ReducePartialScalerMode mode = ReducePartialScalerMode::None;
    // AccumulateViaAdd only: 0/1 mask tile and number of valid lanes in the final reduce-axis tile.
    // These fields are deliberately separate from mode so the existing ReduceTile partial-scaler API
    // (including its single-tile OnlyTile form) remains unchanged.
    uint32_t mask_tile_idx = 0;
    uint32_t valid_reduce_dim_elements = 0;

    static constexpr ReducePartialScaler none() { return {ReducePartialScalerMode::None, 0, 0}; }
    static constexpr ReducePartialScaler last_tile() { return {ReducePartialScalerMode::LastTile, 0, 0}; }
    static constexpr ReducePartialScaler only_tile() { return {ReducePartialScalerMode::OnlyTile, 0, 0}; }
    static constexpr ReducePartialScaler partial_mask(uint32_t valid, uint32_t mask_idx = 0) {
        return {ReducePartialScalerMode::None, mask_idx, valid};
    }

    constexpr bool uses_partial() const { return mode != ReducePartialScalerMode::None; }
    constexpr uint32_t scaler_tile_count() const { return mode == ReducePartialScalerMode::LastTile ? 2 : 1; }
    constexpr uint32_t partial_scaler_idx() const { return mode == ReducePartialScalerMode::LastTile ? 1 : 0; }
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
    uint32_t dst_index = 0;  // DST register for ReduceTile accumulation; AccumulateViaAdd requires 0.

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
 * Unsupported combinations (rejected by static_assert in reduce()):
 * - MAX + REDUCE_SCALAR: the running max cannot be reproduced by the copy_tile reload.
 * - MAX + REDUCE_ROW on Quasar: the reload needs a within-16x16-face transpose that
 *   copy_tile_to_dst_init_short asserts against on Quasar.
 *
 * NOTE on ReducePartialScaler: a partial scaler applies to the last reduce-dim tile of
 * EACH reduce() call, not of the whole accumulated reduction. Combining the two is only
 * correct when every chunk's last tile is genuinely partial. For a streaming reduce where
 * only the final block is short, pass the partial scaler on the LAST call only and
 * ReducePartialScaler::none() on the others.
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
    // AccumulateViaAdd only. CopySeedPairs is valid for both Default and UnpackToDestFp32
    // accumulator CBs; FoldViaAdd is valid only for Default accumulator CBs.
    AccumulateReloadMode reload = AccumulateReloadMode::CopySeedPairs;
    uint32_t iteration = 0;
    // AccumulateViaAdd keeps a raw partial-sum tile between calls. Only the last call performs
    // the within-tile collapse and post-reduce operation.
    bool last = false;

    explicit constexpr Accumulate(AccumulationConfig cfg, uint32_t iter = 0, bool lst = false) :
        config(cfg), iteration(iter), last(lst) {}
    explicit constexpr Accumulate(uint32_t cb, uint32_t iter = 0, bool lst = false) :
        config{cb, 0}, iteration(iter), last(lst) {}

    // Factory for concise call sites
    static constexpr Accumulate at(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }

    // Mark the final AccumulateViaAdd chunk, which collapses within the tile and writes the final output.
    // ReduceTile ignores this flag because each of its chunks is already reduced before accumulation.
    static constexpr Accumulate at_last(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter, true);
    }

    constexpr Accumulate with_reload(AccumulateReloadMode mode) const {
        Accumulate result = *this;
        result.reload = mode;
        return result;
    }

    // Convenience: check if this is first iteration (skip reload)
    constexpr bool is_first() const { return iteration == 0; }
    constexpr bool is_last() const { return last; }
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
 * @tparam input_dfb_id Input DataflowBuffer ID containing tiles to reduce (compile-time CB id)
 * @tparam scaler_dfb_id DataflowBuffer ID containing scaler tile (compile-time CB id)
 * @tparam output_dfb_id Output DataflowBuffer ID for reduced tiles (compile-time CB id)
 *                       The input/output formats are deduced from these CB ids
 *                       (unpack_src_format / pack_dst_format), so Int32 MAX and SUM are routed to
 *                       the SFPU path automatically (Int32 has no FPU support).
 *                       Other formats use FPU/GMPOOL. Only REDUCE_ROW/REDUCE_COL Int32 MAX/SUM on
 *                       SFPU; MIN dispatched via reduce_{h,w}_neg.cpp (SFPU vs FPU branch).
 * @tparam input_policy Input handling policy (default: WaitAndPopPerTile - streaming mode)
 * @tparam reconfig_mode Data format reconfiguration mode (default: INPUT_AND_OUTPUT)
 * @tparam fp32_mode Float32 precision mode (default: Fast). Accurate routes Float32 SUM through
 *                   the SFPU for full-fp32 accumulation; see ReduceFp32Mode.
 *
 * @param input_block_shape Tile grid dimensions (rows x cols x batches)
 *              Use ReduceInputBlockShape::of(r, c, b), ::row(c), ::col(r), or ::single()
 * @param input_memory_layout Tile memory layout specification for NoWaitNoPop/WaitUpfrontNoPop policies (default:
 * contiguous) Use ReduceInputMemoryLayout::with_row_stride(stride) for custom row spacing. Only used when input_policy
 * is NoWaitNoPop or WaitUpfrontNoPop.
 * @param accumulate Accumulation configuration (default: NoAccumulation)
 * @param post_reduce_op Callback after each reduction (default: NoOp)
 * @param partial_scaler Partial-scaler selector for non-tile-aligned reduce
 *        dimensions (default: ReducePartialScaler::none()). Use last_tile()
 *        when the reader emits [full, partial], or only_tile() when it emits
 *        one partial scaler for a single-tile reduction axis.
 *        Not supported for REDUCE_SCALAR or the Int32 SFPU reduce path.
 *
 * @example
 *   // Reduce entire HxW grid to single tile (REDUCE_SCALAR)
 *   compute_kernel_lib::reduce<SUM, REDUCE_SCALAR, dfb_in, dfb_scaler, dfb_out>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * @example
 *   // Reduce each row (W dimension) - output has Ht tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_ROW, dfb_in, dfb_scaler, dfb_out>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * @example
 *   // Reduce each column (H dimension) - output has Wt tiles per batch
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL, dfb_in, dfb_scaler, dfb_out>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * @example
 *   // Reduce type and dimension specified with explicit namespace
 *   compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_SCALAR, dfb_in, dfb_scaler, dfb_out>(
 *       compute_kernel_lib::ReduceInputBlockShape::single());
 *
 * @example
 *   // NoWaitNoPop policy: caller manages wait/pop externally
 *   // Use cases: (1) custom stride between rows, (2) sharded DFB mapped to tensor with data reuse
 *   compute_kernel_lib::reduce<
 *       SUM, REDUCE_ROW, dfb_in, dfb_scaler, dfb_out, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
 *       compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(input_stride));
 *
 * @example
 *   // WaitUpfrontNoPop policy: tiles persist for reuse (ideal for softmax pattern)
 *   // Library waits for tiles internally, but does NOT pop - tiles remain for subsequent ops
 *   compute_kernel_lib::reduce<
 *       MAX, REDUCE_ROW, dfb_values, dfb_scaler, dfb_max,
 *       compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt));
 *   // dfb_values tiles still available for sub_exp_block_bcast_cols_inplace()
 *
 * @example
 *   // BulkWaitBulkPop policy (bulk wait/pop - optimal for performance)
 *   // Library waits for all Wt tiles per row, processes them with indexed access, then pops all Wt tiles
 *   compute_kernel_lib::reduce<
 *       SUM, REDUCE_ROW, dfb_in, dfb_scaler, dfb_out, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
 *
 * @example
 *   // Post-reduce operation: softmax pattern with recip_tile after SUM reduce
 *   compute_kernel_lib::reduce<
 *       SUM, REDUCE_ROW, dfb_exps, dfb_scaler, dfb_out, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
 *       compute_kernel_lib::ReduceInputBlockShape::row(Wt),
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
 *   compute_kernel_lib::reduce<SUM, REDUCE_COL, dfb_in, dfb_scaler, dfb_out>(
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt),
 *       compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
 *       NoAccumulation{},
 *       [](uint32_t dst_idx) {
 *           recip_tile_init();
 *           recip_tile(dst_idx);
 *       });
 */
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast,
    ReduceAlgorithm algorithm = ReduceAlgorithm::Auto,
    typename AccumulateT = NoAccumulation,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{},
    ReducePartialScaler partial_scaler = ReducePartialScaler::none());

/**
 * @brief Mean reduction implemented as a SUM followed by an explicit caller-supplied 1/N.
 *
 * Unlike reduce<AVG>, n_reduced may describe a logical reduction containing partial tiles or an
 * otherwise caller-defined element count. With cross-call Accumulate, n_reduced must cover the
 * whole logical reduction: AccumulateViaAdd applies the normalization only on the at_last() call.
 */
template <
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast,
    ReduceAlgorithm algorithm = ReduceAlgorithm::AccumulateViaAdd,
    typename AccumulateT = NoAccumulation>
ALWI void reduce_mean(
    ReduceInputBlockShape input_block_shape,
    uint32_t n_reduced,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    ReducePartialScaler partial_scaler = ReducePartialScaler::none());

// Compatibility overload for the pre-Accumulate API, where the fourth argument was the partial scaler.
template <
    ReduceDim reduce_dim,
    uint32_t input_dfb_id,
    uint32_t scaler_dfb_id,
    uint32_t output_dfb_id,
    ReduceInputPolicy input_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast,
    ReduceAlgorithm algorithm = ReduceAlgorithm::AccumulateViaAdd>
ALWI void reduce_mean(
    ReduceInputBlockShape input_block_shape,
    uint32_t n_reduced,
    ReduceInputMemoryLayout input_memory_layout,
    ReducePartialScaler partial_scaler);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl"
