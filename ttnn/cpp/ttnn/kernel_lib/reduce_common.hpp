// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tt-metal main moved this to reduce_helpers_common.hpp. When that file is on
// the JIT include path, defer to it so reduce_uses_matmul isn't redefined.
#if defined(__has_include)
#if __has_include("ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp")
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#define COMPUTE_KERNEL_LIB_REDUCE_COMMON_USES_TTMETAL_SPLIT 1
#endif
#endif

#ifndef COMPUTE_KERNEL_LIB_REDUCE_COMMON_USES_TTMETAL_SPLIT

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

#endif  // COMPUTE_KERNEL_LIB_REDUCE_COMMON_USES_TTMETAL_SPLIT
