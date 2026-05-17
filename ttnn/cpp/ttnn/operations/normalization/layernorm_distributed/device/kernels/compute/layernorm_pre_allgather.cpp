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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_14;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;  // x**2

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * x**2
         *
         * Migrated: same-CB BinaryFpu(Mul) over Wt tiles with chain BlockSize=blk.
         * CumulativeWaitNoPop matches the original cb_wait_front(cb_inp, wt+blk)
         * per-iter grow without popping (cb_inp is reused by the sum(x) reduce
         * below — BulkWaitBulkPop there pops the Wt tiles). same_cb dedup means
         * only one wait fires per outer iter even though CbA==CbB==cb_inp.
         * Output packs to absolute idx via PackTile BlockIter +
         * UpfrontReservePushAtEnd (chain reserves Wt up-front, packs sequentially,
         * pushes Wt at end).
         */
        compute_kernel_lib::eltwise_chain<blk>(
            Wt,
            compute_kernel_lib::BinaryFpu<
                cb_inp,
                cb_inp,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::CopyTilePolicy::CumulativeWaitNoPop,
                compute_kernel_lib::CopyTilePolicy::CumulativeWaitNoPop,
                compute_kernel_lib::CbIndexMode::BlockIter,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::CbIndexMode::BlockIter>{},
            compute_kernel_lib::PackTile<
                cb_x2,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::PackTilePolicy::UpfrontReservePushAtEnd,
                compute_kernel_lib::PackTileIndexMode::BlockIter,
                compute_kernel_lib::PackTileReconfig::Output>{});
        /*
         * sum(x**2)
         */

        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        // Bulk mode for optimal performance
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_x2, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        /*
         * sum(x)
         */
        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        // Bulk mode for optimal performance
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_inp, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
    }
    cb_pop_front(cb_reduce, 1);
}
