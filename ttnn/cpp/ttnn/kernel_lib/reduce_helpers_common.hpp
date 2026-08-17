// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"

/**
 * @brief Float32 reduce precision mode.
 *
 * Fast keeps fp32 on the FPU/GMPOOL path (inputs truncated to tf32 — faster, lossy); Accurate
 * routes fp32 SUM through the SFPU for full-fp32 accumulation (accurate ttnn.mean). Only affects
 * Float32 SUM; Int32 always uses the SFPU regardless of this mode.
 */
enum class ReduceFp32Mode : uint8_t { Fast, Accurate };

/**
 * @brief Determines whether a reduce operation should use the SFPU path.
 *
 * Int32 MAX, MIN and SUM on REDUCE_ROW/COL use SFPU (GMPOOL/matmul have no Int32 support).
 * Int32 REDUCE_SCALAR is unsupported (no SFPU scalar primitive); the host decomposes an
 * Int32 HW reduce into a W-then-H two-step (see reduce_op.cpp use_two_step_hw_sfpu_reduce).
 * Int32 MIN drives the LLK MIN reduce directly, instead of the -MAX(-x) reduce_{h,w}_neg path that FPU MIN uses.
 *
 * Float32 SUM additionally opts into the SFPU path when the caller passes ReduceFp32Mode::Accurate
 * (accurate ttnn.mean); the host threads that mode in from the kernel's compile-time args.
 */
template <
    ckernel::PoolType pool_type,
    ckernel::ReduceDim reduce_dim,
    DataFormat data_format,
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast>
constexpr bool is_sfpu_reduce_path() {
    if constexpr (
        pool_type != ckernel::PoolType::MAX && pool_type != ckernel::PoolType::SUM &&
        pool_type != ckernel::PoolType::MIN) {
        return false;
    }
    if constexpr (data_format != DataFormat::Int32) {
        // Float32 opts into the SFPU path only in Accurate mode, and only for SUM (accurate ttnn.mean,
        // which the host lowers to SUM + a 1/N post-mul). Everything else non-Int32 stays on the FPU.
        if constexpr (
            fp32_mode != ReduceFp32Mode::Accurate || data_format != DataFormat::Float32 ||
            pool_type != ckernel::PoolType::SUM) {
            return false;
        }
    }
    if constexpr (reduce_dim == ckernel::ReduceDim::REDUCE_SCALAR) {
        return false;
    }
    return reduce_dim == ckernel::ReduceDim::REDUCE_ROW || reduce_dim == ckernel::ReduceDim::REDUCE_COL;
}

/**
 * @brief Whether the FPU reduce path swaps SrcA/SrcB operands.
 *
 * REDUCE_ROW SUM/AVG uses matmul with scaler in SrcA and data in SrcB (the opposite of the
 * default data→SrcA, scaler→SrcB ordering). This does not apply to MAX (which uses GMPOOL)
 * or to the SFPU path (Int32), which bypasses matmul entirely.
 */
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, bool is_sfpu>
constexpr bool reduce_swaps_operands() {
    return (reduce_dim == ckernel::ReduceDim::REDUCE_ROW) && (pool_type != ckernel::PoolType::MAX) && !is_sfpu;
}
