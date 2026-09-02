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
enum class ReduceFp32Mode : std::uint8_t { Fast, Accurate };

namespace compute_kernel_lib {

/** Concrete input synchronization policy selected by the host planner. */
enum class ReduceInputPolicy {
    WaitAndPopPerTile,
    BulkWaitBulkPop,
    WaitUpfrontNoPop,
    NoWaitNoPop,
    ChunkedWaitChunkedPop,
};

/** Concrete reduction datapath selected by the host planner. */
enum class ReduceAlgorithm { ReduceTile, AccumulateViaAdd };

enum class ReduceWithinTile { Collapse, Skip };

enum class AccumulateReloadMode { FoldViaAdd, CopySeedPairs, CopySeedUniform, CopySeedSfpuAdd, CopySeedZeroPair };

}  // namespace compute_kernel_lib
