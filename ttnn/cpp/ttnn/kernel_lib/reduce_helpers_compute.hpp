// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"
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
 * Call compute_kernel_hw_startup(cb_in, cb_scaler, cb_out) exactly once at the
 * start of your kernel before using. Do NOT re-call it later (and never inside
 * a loop) — re-running mid-kernel can race the compute pipeline and produce
 * undefined behavior.
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
// Algorithm - which datapath implements the reduce
// =============================================================================

/**
 * @brief Which datapath implements the reduce.
 *
 * - Auto (default): pick the implementation automatically. For now this always resolves to ReduceTile;
 *   a cost heuristic (reduced tiles-per-output vs reduce dim, DEST width, input policy, ...) will choose
 *   between the paths later. Callers should prefer Auto and let the library decide.
 *
 * - ReduceTile: the standard datapath — FPU matmul-with-ones (reduce_tile) per input tile, or the SFPU
 *   fold for Int32. Handles EVERY configuration (all pool types, partial / non-tile-aligned reduce dims
 *   via the scaler, cross-call accumulation, all input policies).
 *
 * - AccumulateViaAdd: sum the reduce-dim tiles into ONE DST register with pairwise FPU add_tiles(acc_to_dest),
 *   then finalize within the tile on the SFPU (sfpu_reduce) and, for AVG, apply 1/N with a single SFPU
 *   scalar-multiply. One DST register per output tile, so it handles an arbitrary block without the
 *   REDUCE_COL DST/chunk limit; it wins for wide reduces (many tiles per output) and is more accurate for
 *   AVG / scalar. RESTRICTED — guarded by static_assert / ASSERT in reduce():
 *     - SUM or AVG only (an additive accumulate cannot express MAX/MIN),
 *     - float only (no Int32),
 *     - BulkWaitBulkPop (resident block, indexed) or WaitAndPopPerTile (streaming: DST is the accumulator,
 *       so only ~2 input tiles resident at a time — contiguous row/scalar, aligned only).
 *   PARTIAL (non-tile-aligned) reduce dims are supported standalone (NoAccumulation), ROW/COL only, under
 *   BulkWaitBulkPop: the last reduce-dim tile is folded in with a masked accumulating broadcast-mul and the
 *   mean divides by the true element count. The scaler CB is otherwise unused (1/N is computed internally).
 *   Cross-call Accumulate (CB accumulator across reduce() calls) IS supported: the accumulator CB holds the
 *   RAW partial-sum tile (not a reduced tile), each chunk folds it into the pairwise add NATIVELY (no
 *   binary_dest_reuse) via a parity rule, and sfpu_reduce finalizes only on the last chunk (Accumulate::at_last).
 *   Accumulate is further restricted to SUM (the internal AVG 1/N is per-call and cannot span chunks — for a
 *   cross-chunk mean use SUM + a 1/N post_reduce_op on the last chunk), BulkWaitBulkPop, and tile-aligned (no
 *   partial).
 */
enum class ReduceAlgorithm { Auto, ReduceTile, AccumulateViaAdd };

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
 * TWO scaler tiles into scaler_cb: tile 0 has the full scaler (all positions
 * filled), and tile 1 has the partial scaler (only the valid positions filled,
 * the rest zeroed). The compute kernel must use tile 1 for the *last* tile
 * along the reduce dimension and tile 0 for every other tile.
 *
 * This struct selects which scaler-tile index to use on the last reduce-dim
 * iteration. The default (`none()`) keeps the legacy behavior of using tile 0
 * everywhere. Use `last_tile_at(1)` (or any non-zero idx) to switch to the
 * partial scaler on the last tile.
 *
 * Pair with dataflow_kernel_lib::prepare_partial_reduce_scalers (or
 * calculate_and_prepare_partial_reduce_scalers) on the reader side.
 *
 * REDUCE_SCALAR does not support partial scalers — it applies the scaler
 * twice (row then col), which a single partial tile cannot encode. The
 * runtime asserts that REDUCE_SCALAR callers pass none().
 *
 * Usage:
 *   constexpr auto partial = has_partial
 *       ? ReducePartialScaler::last_tile_at(1)
 *       : ReducePartialScaler::none();
 *   reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, shape, ..., partial);
 */
struct ReducePartialScaler {
    // ReduceTile: scaler-tile index to use for the LAST reduce-dim iteration. 0 = no partial (use tile 0
    // everywhere); >0 = index of the partial scaler tile.
    // AccumulateViaAdd: index of the 0/1 MASK tile in the scaler CB (may be 0).
    uint32_t last_tile_scaler_idx = 0;
    // AccumulateViaAdd only: valid reduce-dim elements in the LAST tile (1..31). >0 signals a partial
    // reduce and gives the true count for the mean's 1/N. 0 = tile-aligned (unused by ReduceTile).
    uint32_t valid_reduce_dim_elements = 0;

    static constexpr ReducePartialScaler none() { return {0, 0}; }
    static constexpr ReducePartialScaler last_tile_at(uint32_t idx = 1) { return {idx, 0}; }
    // AccumulateViaAdd: `valid` real elements in the last reduce-dim tile; 0/1 mask tile at scaler-CB index
    // `mask_idx`.
    static constexpr ReducePartialScaler partial_mask(uint32_t valid, uint32_t mask_idx = 0) {
        return {mask_idx, valid};
    }
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
 * Unsupported combinations (rejected by static_assert in reduce()):
 * - MAX + REDUCE_SCALAR: the running max cannot be reproduced by the copy_tile reload.
 * - MAX + REDUCE_ROW on Quasar: the reload needs a within-16x16-face transpose that
 *   copy_tile_to_dst_init_short asserts against on Quasar.
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
    // AccumulateViaAdd only: marks the LAST chunk. The accumulator CB holds the RAW partial-sum tile, so the
    // within-tile finalize (sfpu_reduce + scaler + post_reduce_op) must run exactly once — on the last chunk,
    // writing the finalized result to the output CB. Non-last chunks write the raw partial sum back to the
    // accumulator CB and skip the finalize. The ReduceTile datapath ignores this flag (it finalizes every
    // chunk, so accumulating REDUCED partials is correct there); only AccumulateViaAdd reads it.
    bool last = false;

    explicit constexpr Accumulate(AccumulationConfig cfg, uint32_t iter = 0, bool lst = false) :
        config(cfg), iteration(iter), last(lst) {}
    explicit constexpr Accumulate(uint32_t cb, uint32_t iter = 0, bool lst = false) :
        config{cb, 0}, iteration(iter), last(lst) {}

    // Factory for concise call sites
    static constexpr Accumulate at(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }
    // AccumulateViaAdd: mark the LAST chunk (finalize within the tile and write to the output CB). Equivalent
    // to at() for the ReduceTile datapath, which finalizes every chunk regardless.
    static constexpr Accumulate at_last(uint32_t cb, uint32_t iter, uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter, /*last=*/true);
    }

    // Convenience: check if this is first iteration (skip reload)
    constexpr bool is_first() const { return iteration == 0; }
    // AccumulateViaAdd: is this the last chunk (finalize + write to output)? See `last`.
    constexpr bool is_last() const { return last; }
};

// NoAccumulation is defined in common_types.hpp (shared with binary_op_helpers).

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

// NoOp is defined in common_types.hpp (shared with binary_op_helpers).

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
 * calling compute_kernel_hw_startup() exactly once at the start of your kernel.
 * Do NOT re-call it later (and never inside a loop) — re-running mid-kernel can
 * race the compute pipeline and produce undefined behavior.
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
 *        dimensions (default: ReducePartialScaler::none()). When set to
 *        last_tile_at(idx), the helper waits for `idx + 1` scaler tiles and
 *        uses scaler tile `idx` for the last reduce-dim iteration. Pair with
 *        dataflow_kernel_lib::prepare_partial_reduce_scalers on the reader.
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

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl"
