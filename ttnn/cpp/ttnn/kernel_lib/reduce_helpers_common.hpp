// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"

/**
 * @brief Determines whether a reduce operation should use the matmul path.
 *
 * On Gen1 (WH/BH), SUM/AVG along REDUCE_ROW uses `matmul_tiles` (with a col-0
 * scaler). It is a hardware-specific perf optimization — the matmul array
 * accumulates the row-sum across columns more efficiently than the reduce LLK.
 * All other combinations use the regular `reduce_tile` LLK (with a row-0 scaler).
 *
 * On Gen2 (Quasar), the matmul-based-reduce LLK API is fundamentally different:
 *   - `llk_math_matmul_init<MathFidelity>` takes (ct_dim, rt_dim) — no
 *     MM_THROTTLE, no CB-id arguments;
 *   - `llk_math_matmul` does not exist (it's `llk_math_matmul_tile`);
 *   - `llk_unpack_reconfig_data_format_srca` for the accumulator-reload path
 *     does not exist either.
 * Rather than maintain a parallel Quasar implementation, we disable the matmul
 * specialization on Quasar and fall back to the standard `reduce_tile` path,
 * which works uniformly for SUM/AVG/MAX × REDUCE_ROW/COL/SCALAR. This loses the
 * Gen1 perf optimization on Quasar but keeps the kernel source single-arch and
 * functionally correct. Both the reader (scaler tile fill) and the compute
 * kernel consult this same predicate, so they stay in sync.
 */
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim>
constexpr bool reduce_uses_matmul() {
#ifdef ARCH_QUASAR
    return false;
#else
    return (pool_type == ckernel::PoolType::SUM || pool_type == ckernel::PoolType::AVG) &&
           reduce_dim == ckernel::ReduceDim::REDUCE_ROW;
#endif
}
