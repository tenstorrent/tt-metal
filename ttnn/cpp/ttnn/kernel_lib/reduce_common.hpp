// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"

/**
 * @brief Determines whether a reduce operation should use the matmul path.
 *
 * SUM/AVG along REDUCE_ROW uses matmul_tiles (col-0 scaler).
 * All other combinations use reduce_tile LLK (row-0 scaler).
 */
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim>
constexpr bool reduce_uses_matmul() {
    return (pool_type == ckernel::PoolType::SUM || pool_type == ckernel::PoolType::AVG) &&
           reduce_dim == ckernel::ReduceDim::REDUCE_ROW;
}
