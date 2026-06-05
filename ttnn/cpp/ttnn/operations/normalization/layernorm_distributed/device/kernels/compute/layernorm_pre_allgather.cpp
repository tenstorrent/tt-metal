// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes larnorm statistics.
 * For layernorm it computes E(x**2) and E(x) and returns them as a two tile wide output tensor containing E(x**2) and
 * E(x) in the left most columns per tile. For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/operations/normalization/kernel_util/compute/pre_add.h"

namespace pre_add = norm::kernel_util::compute::pre_add;

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;                           // x**2
    constexpr uint32_t cb_res = tt::CBIndex::c_5;                          // residual b (unused when !FUSE_PRE_ADD)
    constexpr uint32_t cb_inp = FUSE_PRE_ADD ? tt::CBIndex::c_3 : cb_in0;  // fused a + b, or just a

    if constexpr (FUSE_PRE_ADD) {
        binary_op_init_common(cb_in0, cb_res, cb_inp);
    } else {
        binary_op_init_common(cb_inp, cb_reduce, cb_x2);
    }

    // 2D walk for the squaring chain: outer = Wt/blk subblocks, inner = blk tiles.
    // Total iterations = Wt. Wt is assumed divisible by blk (matches the original
    // loop `for (wt = 0; wt < Wt; wt += blk)` which has no tail handling).
    constexpr auto squaring_shape = compute_kernel_lib::EltwiseShape::of(Wt / blk, blk);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: cb_inp = cb_in0 + cb_res (no-op when !FUSE_PRE_ADD)
        pre_add::one_row<FUSE_PRE_ADD>(cb_in0, cb_res, cb_inp, Wt, blk);

        // x**2 — same-CB FPU mul. cb_inp lifecycle: InputLifecycle::HeldCumulative (chain emits
        // cumulative `cb_wait_front(cb_inp, (i+1)*blk)` per blk-chunk; never pops).
        // The downstream reduce on cb_inp consumes the Wt tiles via BulkWaitBulkPop.
        // cb_x2 lifecycle: OutputLifecycle::Chunked (chain emits cb_reserve_back(blk) +
        // cb_push_back(blk) per chunk; pack writes absolute slots via Block index).
        compute_kernel_lib::square<
            cb_inp,
            cb_x2,
            compute_kernel_lib::InputLifecycle::HeldCumulative,
            compute_kernel_lib::OutputLifecycle::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            compute_kernel_lib::OperandKind::Block>(squaring_shape);

        // sum(x**2) — BulkWaitBulkPop: all Wt tiles already in CB.
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_x2, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // sum(x) — BulkWaitBulkPop pops Wt tiles from cb_inp.
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_inp, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
    }
    cb_pop_front(cb_reduce, 1);
}
