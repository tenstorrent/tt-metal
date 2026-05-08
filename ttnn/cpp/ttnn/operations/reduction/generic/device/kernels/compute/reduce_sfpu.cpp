// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU sibling of reduce.cpp.  Routes the reduction through the SFPU
// (sfpu_reduce + binary_*_int32_tile) instead of the FPU's GMPOOL primitive
// so it can handle integer formats GMPOOL silently zeroes (Int32 -- see
// issue #26726, plan in #43736).
//
// Compile-time defines (set by the host program factory):
//   REDUCE_OP:     ckernel::PoolType::MAX or PoolType::MIN
//                  (Phase 1 scope -- SUM is reserved for follow-up).
//   REDUCE_DIM:    ckernel::ReduceDim::REDUCE_ROW (W axis) or REDUCE_COL (H axis).
//   REDUCE_FORMAT: ckernel::DataFormat::Int32 (Phase 1 scope).
//   REDUCE_NEGATE: when set to 1 the kernel computes `-REDUCE_OP(-x)`,
//                  i.e. negates each input tile before the reduce and the
//                  output tile after the reduce.  The host uses this to
//                  lower MIN to MAX (mirrors reduce_w_neg / reduce_h_neg).
//
// Compile-time args:
//   0: Ht (number of input tiles along H per batch -- per-core for W reduce)
//   1: Wt (number of input tiles along W per batch -- per-core for H reduce)
//   2: NC (number of (N,C) batches; for H reduce it is set to 1 unless width-sharded)
//
// Circular buffers:
//   c_0: input tiles  (filled by the reader)
//   c_2: scaler tile  (pushed by the reader for the FPU path; drained but unused here)
//   c_3: output tiles (consumed by the writer)
//
// All the reduction logic is in compute_kernel_lib::reduce_sfpu, which
// mirrors compute_kernel_lib::reduce so the parameter shape (op x dim x format)
// stays consistent across the FPU and SFPU paths.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.hpp"

#ifndef REDUCE_NEGATE
#define REDUCE_NEGATE 0
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);

    compute_kernel_lib::reduce_sfpu<REDUCE_OP, REDUCE_DIM, REDUCE_FORMAT, /*negate=*/(REDUCE_NEGATE != 0)>(
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));
}
