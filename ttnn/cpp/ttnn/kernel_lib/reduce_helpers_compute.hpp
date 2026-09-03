// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <cstdint>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp"
/**
 * @file reduce_helpers_compute.hpp
 * @brief Single unified reduce function with an explicitly selected datapath
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
 *
 * - ChunkedWaitChunkedPop: Keep the logical output resident in DEST while waiting for and
 *   popping host-planned reduction-axis chunks. The chunk descriptor passed to reduce()
 *   supplies the reduction-axis and output-group tile counts.
 *
 * Output synchronization is independent of the input policy: each output tile is
 * reserved and pushed individually.
 */
// =============================================================================
// Algorithm - which datapath implements the reduce
// =============================================================================

/**
 * @brief Which datapath implements the reduce.
 *
 * - ReduceTile (default): the standard datapath — FPU matmul-with-ones (reduce_tile) per input tile, or the SFPU
 *   fold for Int32. Handles EVERY configuration (all pool types, partial / non-tile-aligned reduce dims
 *   via the scaler, cross-call accumulation, all input policies).
 *
 * - AccumulateViaAdd: sum the reduce-dim tiles into ONE DST register with pairwise FPU add_tiles(acc_to_dest),
 *   then finalize within the tile on the SFPU (sfpu_reduce) and, for AVG, apply the compile-time
 *   1/reduce_factor with a single SFPU scalar-multiply. One DST register per output tile, so it handles an
 *   arbitrary block without the REDUCE_COL DST/chunk limit; it wins for wide reduces (many tiles per output)
 *   and is more accurate for AVG / scalar.
 *   Boots like every reduce — compute_kernel_hw_startup(cb_in, cb_scaler, cb_out) once at kernel start (see the
 *   file-level note); reduce() runs no heavy per-call hw_configure — per call it does only light format reconfig
 *   (per reconfig_mode) + the SFPU-macro load, exactly like ReduceTile relies on boot + light reduce_init.
 *   RESTRICTED — guarded by static_assert / ASSERT in reduce():
 *     - SUM or AVG. AVG uses the caller-supplied compile-time reduce_factor, so standalone, partial,
 *       cross-chunk, sharded, and uneven means all use the same reduce<AVG> entry point. MAX/MIN are not
 *       expressible via additive accumulate,
 *     - float only (no Int32),
 *     - all input policies except WaitAndPopPerTile + COL. WaitAndPopPerTile streams contiguous ROW/SCALAR;
 *       ChunkedWaitChunkedPop also supports COL by retaining an output-column group in DEST while row chunks
 *       arrive (the reduction-axis chunk must contain at least two tiles),
 *   PARTIAL (non-tile-aligned) reduce dims are supported standalone (NoAccumulation), ROW/COL only. The last
 *   reduce-dim tile is folded in with a masked accumulating broadcast-mul so padding contributes 0. The scaler
 *   CB is otherwise unused. For partial AVG, reduce_factor is the true element count:
 *   full_tiles*32 + valid_elems_in_last_tile.
 *   Cross-call Accumulate (CB accumulator across reduce() calls) IS supported: the accumulator CB holds the
 *   RAW partial-sum tile (not a reduced tile), each later call copy-seeds or folds it into DEST, and sfpu_reduce
 *   finalizes only on the last call (Accumulate::at_last). Resident and streamed/chunked input are supported.
 *   PARTIAL (ROW/COL) composes with Accumulate — the masked last tile folds into each chunk's sum via
 *   fold_partial_last. CopySeedZeroPair may be used at the same time: the zero tile immediately follows the
 *   partial mask/scaler representation in the auxiliary CB. A cross-chunk AVG uses the GRAND-TOTAL
 *   reduce_factor; raw partial sums remain unscaled until the last chunk finalizes.
 */
/**
 * @brief Whether AccumulateViaAdd runs the WITHIN-TILE collapse at the end of the reduction.
 *
 * CONTRACT — Skip is valid only with `ReduceAlgorithm::AccumulateViaAdd` and PoolType::SUM or PoolType::AVG.
 * `ReduceAlgorithm::ReduceTile` + `Skip` does NOT compile: request AccumulateViaAdd explicitly. MAX/MIN are
 * rejected because the accumulate datapath's cross-tile step is an
 * add, so "skip the collapse" has no meaning for those reductions.
 *
 * AccumulateViaAdd is two distinct steps: (1) sum the reduce-dim TILES into one DST register with pairwise
 * add_tiles(acc_to_dest), then (2) collapse the 32 lanes INSIDE that tile with sfpu_reduce. Step (2) is only
 * needed when the inputs carry real data across the reduce axis.
 *
 * Skip is for inputs that are ALREADY collapsed on that axis — the classic case is summing per-core PARTIALS
 * that each came out of an earlier reduce<..., REDUCE_ROW> and are therefore column-0-valid. The sum of
 * column-0-valid tiles is column-0-valid, so the sfpu_reduce would be a no-op over 31 garbage lanes. Skipping
 * it removes an SFPU pass per output tile.
 *
 * With Skip, DST holds the raw cross-tile SUM. AVG applies the caller's 1/reduce_factor to that sum, allowing a
 * cross-core mean combine to divide by its number of contributors without an unnecessary lane collapse. The
 * independent `post_reduce_op` runs after that normalization, so eps-add or rsqrt composes normally.
 *
 * SFPU STATE: Skip omits sfpu_reduce. AVG still arms the scalar multiply when reduce_factor != 1, but an SFPU
 * `post_reduce_op` should follow the normal contract and run its own <op>_tile_init. Under Collapse a post-op
 * may instead free-ride on the finalize's sfpu_reduce init.
 *
 * PARTIAL IS REJECTED (asserted): `use_partial` means the last tile needs a lane mask along the reduce axis.
 * Skip's inputs are already collapsed on that axis, so there is nothing for a 0/1 lane mask to mask.
 *
 * ReduceTile cannot express Skip: there the reduce_tile matmul-with-ones IS the collapse, so there is nothing
 * to skip (asserted).
 */
/*
 * SUMMING TILES THAT ARE ALREADY REDUCED (the cross-core combine) — two ways, NEITHER of which
 * needs a tile of zeros.
 *
 * The situation: each core produced a partial with reduce<SUM, REDUCE_ROW> (so it is column-0-valid),
 * the partials have been gathered, and you now want their sum. It is tempting to reach for a BinaryFpu
 * whose B operand is a zero tile, because BinaryFpu takes TWO CB inputs and you only have one stream.
 * Do not — the zero tile has to be filled, and that fill is never free.
 *
 * (1) THIS HELPER. `reduce<SUM, ..., ReduceAlgorithm::AccumulateViaAdd, ..., ReduceWithinTile::Skip>`
 *     sums the tiles and skips the within-tile collapse they do not need. Prefer this when the shape
 *     fits a ReduceInputBlockShape.
 *
 * (2) A RAW PAIRWISE CHAIN, when you want the fold inline in an eltwise_chain. Point BOTH operands at
 *     the SAME CB and offset the second by half the run: step i then computes `x[i] + x[i + N/2]`, and
 *     DestAccumulation sums those N/2 pair-sums into one DEST — the N-way total, in N/2 FPU steps,
 *     with no identity operand:
 *
 *         constexpr uint32_t HALF = N / 2;              // N = tiles to sum; N must be EVEN
 *         ckl::eltwise_chain(
 *             ckl::IterationShape::tiles(HALF),
 *             ckl::BinaryFpu<
 *                 ckl::input(cb_in, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block),
 *                 ckl::input(cb_in, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block,
 *                            ckl::TileOffset::Set),
 *                 ckl::BinaryFpuOp::Add,
 *                 ckl::BroadcastDim::None,
 *                 ckl::Dst::D0,
 *                 ckl::DestAccumulation::WholeShape>{0, HALF},   // <- the two operand BASES
 *             ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter,
 *                                       ckl::DataFormatReconfig::Enabled, ckl::PackRelu::Disabled,
 *                                       ckl::L1Accumulation::Disabled,
 *                                       ckl::DestAccumulation::WholeShape)>{});
 *
 *     `{0, HALF}` are the A and B operand base offsets — that brace pair is the whole trick, and it is
 *     the same idiom the `eltwise_l1_vs_dest_accumulate` example measures. `tiles(...)` is one
 *     contiguous shape, so the accumulation scope is WholeShape (PerRow is rejected there); for a 2D
 *     walk use `grid(H, W)` with `TileOffset::Strided` and a `StridedTileRange{base, row_stride}` per
 *     operand. ODD N does not tile into halves — fall back to (1), or handle the leftover separately.
 */

/**
 * @brief How AccumulateViaAdd's cross-call Accumulate folds the running accumulator (cb_accumulator) with a
 * later chunk's new tiles. Only affects AccumulateViaAdd + Accumulate later chunks (ignored for the first
 * chunk / NoAccumulation / ReduceTile).
 *
 * CONTRACT: FoldViaAdd reads the accumulator CB through SrcA/SrcB, so it is ONLY valid when that CB is
 * UnpackToDestMode::Default. If the accumulator CB is tagged UnpackToDestMode::UnpackToDestFp32 (a lossless
 * fp32 reload — SrcA/B access is disabled for it, see the numeric-formats docs), FoldViaAdd is INCORRECT; use
 * a CopySeed* mode (reloads via copy_tile, the only sanctioned access for a to-dest CB).
 *
 * - FoldViaAdd: fold the accumulator as an add_tiles SRCB operand (no dest reload). Fastest; Default-acc only.
 * - CopySeedPairs: reload the accumulator into DST via copy_tile, then add the new tiles — pairwise add_tiles
 *   for the bulk (2 tiles/op) + one DEST-reuse add for an odd leftover. Safe for any accumulator CB.
 * - CopySeedUniform: reload via copy_tile, then add every new tile via a DEST-reuse add (1 tile/op). Safe;
 *   simplest; slower bulk. (Kept mainly for the bake-off; CopySeedPairs dominates it.)
 * - CopySeedSfpuAdd: sum the new tiles into DST[0] with pure pairwise add_tiles (fresh DST, full fp32, no
 *   DEST-reuse truncation), reload the accumulator into DST[1] via copy_tile, then SFPU-add DST[0] += DST[1].
 *   Safe; MOST accurate (no TF32 round-trip anywhere), at the cost of one extra copy_tile + SFPU add per
 *   output. WH/BH only (add_binary_tile is not available on Quasar).
 * - CopySeedZeroPair: copy_tile-reload the accumulator into DST[0], then add the new tiles in pairs; the odd
 *   leftover is paired with a ZERO tile (in scaler_dfb) via an acc_to_dest add_tiles, which keeps the running
 *   sum in fp32 DST (no DEST-reuse TF32 truncation) with NO SFPU op. Aims for CopySeedSfpuAdd accuracy at
 *   CopySeedPairs speed. With no partial, the zero is scaler_dfb[0]. With a partial mask/scaler representation,
 *   the zero immediately follows it (for AccumulateViaAdd's mask-only layout: mask[0], zero[1]).
 */
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
    std::uint32_t row_stride = 0;  // 0 = auto-detect from Wt (contiguous row-major)

    explicit constexpr ReduceInputMemoryLayout() = default;
    explicit constexpr ReduceInputMemoryLayout(std::uint32_t row) : row_stride(row) {}

    static constexpr ReduceInputMemoryLayout contiguous() { return ReduceInputMemoryLayout(); }
    static constexpr ReduceInputMemoryLayout with_row_stride(std::uint32_t s) { return ReduceInputMemoryLayout(s); }
};

/**
 * @brief Host-planned input synchronization geometry.
 *
 * A zero field requests the existing automatic/default geometry. Non-zero values are required
 * by ReduceInputPolicy::ChunkedWaitChunkedPop. output_tiles is normally one, except REDUCE_COL
 * where it is the number of independent columns held in DEST together.
 */
struct ReduceInputChunk {
    std::uint32_t reduce_axis_tiles = 0;
    std::uint32_t output_tiles = 0;

    static constexpr ReduceInputChunk automatic() { return {}; }
    static constexpr ReduceInputChunk of(std::uint32_t reduce_tiles, std::uint32_t outputs = 1) {
        return {reduce_tiles, outputs};
    }
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
    std::uint32_t rows;
    std::uint32_t cols;
    std::uint32_t batches;

    static constexpr ReduceInputBlockShape of(std::uint32_t r, std::uint32_t c, std::uint32_t b = 1) {
        return {r, c, b};
    }
    static constexpr ReduceInputBlockShape single() { return {1, 1, 1}; }
    static constexpr ReduceInputBlockShape row(std::uint32_t c, std::uint32_t b = 1) { return {1, c, b}; }
    static constexpr ReduceInputBlockShape col(std::uint32_t r, std::uint32_t b = 1) { return {r, 1, b}; }
};

/**
 * ReducePartialMode is the planner-produced partial-edge contract. Kernel
 * callers pass the mode directly; reduce() owns the auxiliary-CB layout needed
 * to implement it.
 *
 * REDUCE_SCALAR does not support a partial mode: ReduceTile applies its scaler
 * twice (row then col), while one AccumulateViaAdd row/column mask cannot encode
 * a 2-D partial corner.
 */

/**
 * @brief Configuration for accumulation-style reductions
 *
 * Holds the static configuration for accumulation (CB and DST index).
 * Does not hold iteration state - that's provided via Accumulate wrapper.
 */
struct AccumulationConfig {
    // CB holding the running accumulator tile across reduce() iterations; see Accumulate below.
    std::uint32_t cb_accumulator = 0;
    std::uint32_t dst_index = 0;  // DST register for accumulation (default: 0)

    static constexpr AccumulationConfig with_cb(std::uint32_t cb, std::uint32_t dst = 0) { return {cb, dst}; }
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
 * - iteration > 0: reload from the accumulator CB before reducing. The CB must expose one tile
 *   per output, in output order; REDUCE_COL reloads and pops a complete DEST chunk at a time.
 *
 * Unsupported combinations (rejected by static_assert in reduce()):
 * - MAX + REDUCE_SCALAR: the running max cannot be reproduced by the copy_tile reload.
 * - MAX + REDUCE_ROW on Quasar: the reload needs a within-16x16-face transpose that
 *   copy_tile_to_dst_init_short asserts against on Quasar.
 *
 * NOTE on ReducePartialMode: partial metadata applies to the last reduce-dim tile of
 * EACH reduce() call, not of the whole accumulated reduction. When only the final chunk
 * is short, pass Scaler (ReduceTile) or Mask (AccumulateViaAdd) on that call only and
 * None on the others.
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
    // AccumulateViaAdd: how a later call folds the accumulator with its new tiles. Default is safe for any
    // accumulator CB, including UnpackToDestFp32. Indexed input supports every mode; streamed/grouped input
    // uses its safe copy seed and additionally honors CopySeedZeroPair for odd tiles after DEST is seeded.
    AccumulateReloadMode reload = AccumulateReloadMode::CopySeedPairs;
    std::uint32_t iteration = 0;
    // AccumulateViaAdd only: marks the LAST chunk. The accumulator CB holds the RAW partial-sum tile, so the
    // within-tile finalize (sfpu_reduce + scaler + post_reduce_op) must run exactly once — on the last chunk,
    // writing the finalized result to the output CB. Non-last chunks write the raw partial sum back to the
    // accumulator CB and skip the finalize. The ReduceTile datapath ignores this flag (it finalizes every
    // chunk, so accumulating REDUCED partials is correct there); only AccumulateViaAdd reads it.
    bool last = false;

    explicit constexpr Accumulate(AccumulationConfig cfg, std::uint32_t iter = 0, bool lst = false) :
        config(cfg), iteration(iter), last(lst) {}
    explicit constexpr Accumulate(std::uint32_t cb, std::uint32_t iter = 0, bool lst = false) :
        config{cb, 0}, iteration(iter), last(lst) {}

    // Factory for concise call sites
    static constexpr Accumulate at(std::uint32_t cb, std::uint32_t iter, std::uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter);
    }
    // AccumulateViaAdd: mark the LAST chunk (finalize within the tile and write to the output CB). Equivalent
    // to at() for the ReduceTile datapath, which finalizes every chunk regardless.
    static constexpr Accumulate at_last(std::uint32_t cb, std::uint32_t iter, std::uint32_t dst = 0) {
        return Accumulate(AccumulationConfig{cb, dst}, iter, /*last=*/true);
    }

    // Fluent: select the later-chunk reload strategy (AccumulateViaAdd only). e.g.
    // Accumulate::at(cb, c).with_reload(AccumulateReloadMode::FoldViaAdd).
    constexpr Accumulate with_reload(AccumulateReloadMode m) const {
        Accumulate a = *this;
        a.reload = m;
        return a;
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
struct is_post_reduce_op<T, std::void_t<decltype(std::declval<T>()(std::declval<std::uint32_t>()))>> : std::true_type {
};

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
 *                       (unpack_src_format / pack_dst_format), so Int32 is routed to the SFPU path
 *                       automatically (Int32 has no FPU support). Other formats use FPU/GMPOOL
 *                       unless Accurate fp32 is requested. SFPU covers REDUCE_ROW/REDUCE_COL only;
 *                       fast-mode float/bf16 MIN is dispatched via reduce_{h,w}_neg.cpp.
 * @tparam input_policy Input handling policy (default: WaitAndPopPerTile - streaming mode)
 * @tparam reconfig_mode Data format reconfiguration mode (default: INPUT_AND_OUTPUT)
 * @tparam fp32_mode Float32 precision mode (default: Fast). Accurate routes Float32 through the
 *                   SFPU at full fp32; see ReduceFp32Mode.
 * @tparam reduce_factor Number of elements in an AVG reduction (default: 1). A non-default value is valid only
 *                       for AVG. AVG requires a non-default value on AccumulateViaAdd and SFPU paths; the
 *                       ordinary ReduceTile path may keep 1 because it consumes its reader-provided scaler.
 *                       AccumulateViaAdd evaluates the reciprocal at compile time and applies it after the
 *                       within-tile finalize and before the independent post_reduce_op.
 *
 * @param input_block_shape Tile grid dimensions (rows x cols x batches)
 *              Use ReduceInputBlockShape::of(r, c, b), ::row(c), ::col(r), or ::single()
 * @param input_memory_layout Tile memory layout specification for NoWaitNoPop/WaitUpfrontNoPop policies (default:
 * contiguous) Use ReduceInputMemoryLayout::with_row_stride(stride) for custom row spacing. Only used when input_policy
 * is NoWaitNoPop or WaitUpfrontNoPop.
 * @param accumulate Accumulation configuration (default: NoAccumulation)
 * @param post_reduce_op Callback after each reduction (default: NoOp)
 * @param partial_mode Planner-selected handling for a non-tile-aligned reduce
 *        dimension (default: ReducePartialMode::None). Kernel callers do not
 *        select or address auxiliary tiles.
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
    std::uint32_t input_dfb_id,
    std::uint32_t scaler_dfb_id,
    std::uint32_t output_dfb_id,
    ReduceInputPolicy input_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast,
    ReduceAlgorithm algorithm = ReduceAlgorithm::ReduceTile,
    // within_tile sits AHEAD of the two deduced typename parameters on purpose: PostReduceOp is normally a
    // lambda, whose type cannot be named at the call site, so a trailing within_tile would be unreachable for
    // every caller that passes a post_reduce_op — i.e. exactly the callers Skip is documented for.
    ReduceWithinTile within_tile = ReduceWithinTile::Collapse,
    std::uint32_t reduce_factor = 1,
    typename AccumulateT = NoAccumulation,
    typename PostReduceOp = NoOp>
ALWI void reduce(
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{},
    ReducePartialMode partial_mode = ReducePartialMode::None,
    ReduceInputChunk input_chunk = ReduceInputChunk::automatic());

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl"
