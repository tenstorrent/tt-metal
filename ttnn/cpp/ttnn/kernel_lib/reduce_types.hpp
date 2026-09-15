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
 * Fast keeps fp32 on the FPU/GMPOOL path. Accurate routes supported fp32
 * reductions through the SFPU at full fp32 precision.
 */
enum class ReduceFp32Mode : std::uint8_t { Fast = 0, Accurate = 1 };

namespace ttnn::kernel_lib {

enum class ReducePath : std::uint8_t { Tiled = 0, DenseRowMajor = 1 };

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
    // ReduceTile's partial REDUCE_COL scaler encodes valid rows across row 0
    // of each participating face row.
    FirstRowPerFaceRow = 2,
    Zero = 3,
};

}  // namespace ttnn::kernel_lib

namespace compute_kernel_lib {

/** Concrete input synchronization policy selected by the host planner. */
enum class ReduceInputPolicy : std::uint8_t {
    WaitAndPopPerTile = 0,
    BulkWaitBulkPop = 1,
    WaitUpfrontNoPop = 2,
    NoWaitNoPop = 3,
    ChunkedWaitChunkedPop = 4,
};

/** Concrete reduction datapath selected by the host planner. */
enum class ReduceAlgorithm : std::uint8_t { ReduceTile = 0, AccumulateViaAdd = 1 };

enum class ReduceWithinTile : std::uint8_t { Collapse = 0, Skip = 1 };

enum class ReduceDataFormatReconfigMode : std::uint8_t {
    NONE = 0,
    INPUT = 1,
    OUTPUT = 2,
    INPUT_AND_OUTPUT = 3,
};

enum class AccumulateReloadMode : std::uint8_t {
    FoldViaAdd = 0,
    CopySeedPairs = 1,
    CopySeedUniform = 2,
    CopySeedSfpuAdd = 3,
    CopySeedZeroPair = 4,
};

/** Host-planned treatment of a non-tile-aligned reduction edge. */
enum class ReducePartialMode : std::uint8_t {
    None = 0,
    // ReduceTile uses its ordinary scaler for full tiles and a partial scaler
    // for the last tile along the reduction axis.
    Scaler = 1,
    // AccumulateViaAdd masks the last tile before folding it into DEST.
    Mask = 2,
};

}  // namespace compute_kernel_lib
