// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Shared writer-side setup for the fused Wan2.2 distributed RMSNorm op.
 *
 * The WRITER (both the is_tp_1 drain-only writer and the forwarder-AG worker
 * writer) always populates the compute kernel's reduce-scalar / epsilon CBs and
 * reads the RoPE transformation matrix. Doing this in the writer (rather than
 * the reader) lets the reader issue its first input-tile read immediately, so
 * compute's PRE sum-of-squares starts as early as possible.
 *
 * Both writers call this with their own CB ids / accessor, so the logic lives
 * in exactly one place.
 */

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

// Generate the SUM (pre) and AVG (post, scaled by reduce_factor == 1/H_full)
// reduce scalars + the bcast-col epsilon tile, then — if RoPE is fused — read
// the single transformation-matrix tile into its CB. `tmat_acc` is only
// dereferenced when fuse_rope is true; callers without RoPE may pass any
// TensorAccessor (it is never read).
template <
    uint32_t sum_cb,
    uint32_t avg_cb,
    uint32_t eps_cb,
    uint32_t transmat_cb,
    uint32_t reduce_factor,
    bool fuse_rope,
    typename TMatAccessor>
FORCE_INLINE void dit_rmsnorm_generate_scalars_and_transmat(uint32_t eps_bits, const TMatAccessor& tmat_acc) {
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<sum_cb, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        avg_cb,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        reduce_factor>();
    generate_bcast_col_scalar(CircularBuffer(eps_cb), eps_bits);

    if constexpr (fuse_rope) {
        cb_reserve_back(transmat_cb, 1);
        const uint32_t transmat_wr_ptr = get_write_ptr(transmat_cb);
        noc_async_read_page(0, tmat_acc, transmat_wr_ptr);
        noc_async_read_barrier();
        cb_push_back(transmat_cb, 1);
    }
}
