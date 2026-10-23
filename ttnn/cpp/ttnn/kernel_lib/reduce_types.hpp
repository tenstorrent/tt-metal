// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

enum class ReduceOpMath { SUM, AVG, MAX, MIN, STD, VAR };

enum class ReduceOpDim { H, W, HW };

enum class ReduceOpParallelizationStrategy { MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE_HW };

}  // namespace tt::tt_metal

/**
 * @brief Float32 reduce precision mode.
 *
 * Fast keeps fp32 on the FPU/GMPOOL path (inputs truncated to tf32 — faster, lossy); Accurate
 * routes fp32 through the SFPU at full fp32. Only affects Float32; Int32 and bf16 MIN use the SFPU
 * regardless of this mode.
 */
enum class ReduceFp32Mode : std::uint8_t { Fast = 0, Accurate = 1 };

namespace ttnn::kernel_lib {

// The accumulation behavior of one independently serialized reduce call.
// This is planned on the host; a consuming kernel must not infer it from the
// call's position in a list.
enum class ReduceAccumulationMode : std::uint8_t {
    None = 0,
    Intermediate = 1,
    Final = 2,
};

// Physical pattern written into one reduction auxiliary tile. These describe
// only tile contents; they deliberately carry no reduction algorithm or policy
// semantics so the dataflow helper can be a generic recipe executor.
enum class ReduceAuxiliaryTileType : std::uint8_t {
    FirstRow = 0,
    FirstColumn = 1,
    // Native's partial REDUCE_COL scaler encodes valid rows across row 0
    // of each participating face row.
    FirstRowPerFaceRow = 2,
    Zero = 3,
    // Pushed without being written, for reductions that read no auxiliary tile; contents are unspecified.
    Placeholder = 4,
};

}  // namespace ttnn::kernel_lib

namespace compute_kernel_lib {

/** Auxiliary CB ID for reductions that read no auxiliary tile (the Int32 and accurate-fp32 SFPU paths). */
inline constexpr std::uint32_t REDUCE_NO_AUXILIARY_CB = 0xFF;

/** float32 bits of 1.0f, the reduce() scale that leaves outputs unscaled. */
inline constexpr std::uint32_t REDUCE_UNIT_SCALE_BITS = 0x3F800000;

/** How reduce() waits for and pops input tiles. Output tiles are always reserved and pushed one at a time. */
enum class ReduceInputPolicy : std::uint8_t {
    // Stream input. ReduceAlgorithm::TileSum consumes pairs, so the input CB needs room for two tiles.
    WaitAndPopPerTile = 0,
    // Wait for a block, read it indexed, pop it: Ht*Wt tiles for SCALAR, Wt for ROW, Ht*output_group for COL.
    BulkWaitBulkPop = 1,
    // Wait for all tiles up front and leave them in the CB. Tiles are indexed row-major.
    WaitUpfrontNoPop = 2,
    // The caller owns wait/pop. Tiles are indexed row-major.
    NoWaitNoPop = 3,
};

/**
 * Datapath that implements the reduce.
 *
 * ReduceAlgorithm::Native: reduce_tile per input tile, or the SFPU path for Int32, bf16 MIN and accurate fp32. Supports
 * every pool type, reduce dim and input policy.
 *
 * ReduceAlgorithm::TileSum: float SUM/AVG only. Adds the reduce-dim tiles pairwise into one DST register per output,
 * then collapses within the tile on the SFPU and multiplies by the caller's scale. Not limited by DST capacity
 * along the reduce dim; faster for wide reductions and more accurate for AVG. Streamed REDUCE_COL requires
 * output_group = 1. Across Accumulate calls the accumulator holds the raw sum, which only Accumulate::at_last
 * finalizes, so an AVG scale covers the grand total.
 */
enum class ReduceAlgorithm : std::uint8_t { Native = 0, TileSum = 1 };

/**
 * Whether ReduceAlgorithm::TileSum collapses the summed tile within the tile. Skip is for inputs already collapsed
 * along the reduce dim, e.g. per-core REDUCE_ROW partials. Scaling and post_reduce_op still run; an SFPU
 * post_reduce_op must call its own init. Skip requires ReduceAlgorithm::TileSum, SUM/AVG and ReducePartialMode::None.
 */
enum class ReduceWithinTile : std::uint8_t { Collapse = 0, Skip = 1 };

/**
 * Format reconfiguration reduce() performs on entry. INPUT_AND_OUTPUT is always correct; narrower modes save cycles
 * when the previous op already configured the matching formats. Switches to accumulator or auxiliary tiles are
 * always reconfigured and restored. Quasar output descriptor initialization is required regardless of the mode.
 */
enum class ReduceDataFormatReconfigMode : std::uint8_t {
    NONE = 0,
    INPUT = 1,
    OUTPUT = 2,
    INPUT_AND_OUTPUT = 3,
};

/** How a later ReduceAlgorithm::TileSum + Accumulate call folds the accumulator CB with its new tiles. */
enum class AccumulateReloadMode : std::uint8_t {
    // The accumulator is an add_tiles SrcB operand. Fastest; invalid for an UnpackToDestFp32 accumulator CB.
    FoldViaAdd = 0,
    // copy_tile the accumulator into DST, then add input tiles in pairs and an odd one with the auxiliary zero.
    CopySeedPairs = 1,
    // Sum the input tiles in a fresh DST, copy_tile the accumulator beside it and add on the SFPU. Most accurate;
    // not available on Quasar.
    CopySeedSfpuAdd = 2,
};

/**
 * Treatment of a non-tile-aligned reduction edge, applied to the last reduce-dim tile of each reduce() call.
 * REDUCE_ROW/REDUCE_COL on the FPU paths only. Auxiliary CB layout from auxiliary_tile_offset:
 *   ReduceAlgorithm::Native + Scaler: [full scaler, partial scaler]
 *   ReduceAlgorithm::TileSum + Mask: [mask, zero]
 *   ReduceAlgorithm::TileSum + None: [zero]
 */
enum class ReducePartialMode : std::uint8_t {
    None = 0,
    // ReduceAlgorithm::Native scales the last tile with the partial scaler; a zero scaler also excludes MAX inputs.
    Scaler = 1,
    // ReduceAlgorithm::TileSum masks the last tile before folding it into DST.
    Mask = 2,
};

}  // namespace compute_kernel_lib
