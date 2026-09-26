// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @brief Float32 reduce precision mode.
 *
 * Fast keeps fp32 on the FPU/GMPOOL path (inputs truncated to tf32 — faster, lossy); Accurate
 * routes fp32 through the SFPU at full fp32. Only affects Float32; Int32 always uses the SFPU
 * regardless of this mode.
 */
enum class ReduceFp32Mode : std::uint8_t { Fast = 0, Accurate = 1 };

namespace compute_kernel_lib {

/**
 * Auxiliary CB ID for reductions that read no auxiliary tile (the Int32 and
 * accurate-fp32 SFPU paths).
 */
inline constexpr std::uint32_t REDUCE_NO_AUXILIARY_CB = 0xFF;

enum class ReduceInputPolicy : std::uint8_t {
    WaitAndPopPerTile = 0,
    BulkWaitBulkPop = 1,
    WaitUpfrontNoPop = 2,
    NoWaitNoPop = 3,
};

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

/** Treatment of a non-tile-aligned reduction edge. */
enum class ReducePartialMode : std::uint8_t {
    None = 0,
    // ReduceTile uses its ordinary scaler for full tiles and a partial scaler
    // for the last tile along the reduction axis.
    Scaler = 1,
    // AccumulateViaAdd masks the last tile before folding it into DEST.
    Mask = 2,
};

}  // namespace compute_kernel_lib
