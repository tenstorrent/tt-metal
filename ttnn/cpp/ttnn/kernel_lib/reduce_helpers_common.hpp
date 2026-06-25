// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"

/**
 * @brief Determines whether a reduce operation should use the SFPU path.
 *
 * Int32 MAX and SUM on REDUCE_ROW/COL use SFPU (GMPOOL/matmul have no Int32 support).
 * Int32 REDUCE_SCALAR is unsupported (no SFPU scalar primitive); the host decomposes an
 * Int32 HW reduce into a W-then-H two-step (see reduce_op.cpp use_two_step_hw_sfpu_reduce).
 * MIN is pre-lowered to MAX + negate and dispatched via reduce_{h,w}_neg.
 */
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat data_format>
constexpr bool is_sfpu_reduce_path() {
    if constexpr (pool_type != ckernel::PoolType::MAX && pool_type != ckernel::PoolType::SUM) {
        return false;
    }
    if constexpr (data_format != DataFormat::Int32) {
        return false;
    }
    if constexpr (reduce_dim == ckernel::ReduceDim::REDUCE_SCALAR) {
        return false;
    }
    return reduce_dim == ckernel::ReduceDim::REDUCE_ROW || reduce_dim == ckernel::ReduceDim::REDUCE_COL;
}
