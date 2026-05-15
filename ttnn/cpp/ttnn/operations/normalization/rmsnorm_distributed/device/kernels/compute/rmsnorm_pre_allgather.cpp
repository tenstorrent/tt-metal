// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
 * For rmsnorm we compute E(x**2) and return it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

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
         * Migrated: same-CB BinaryFpu(Mul) with cumulative wait + chain-owned pop, plus
         * per-tile pack into cb_x2. CumulativeWaitPopAtEnd matches the inner-loop
         * cb_wait_front(cb_inp, wt + blk) / per-iter pack pattern (blk == 1 in the
         * program factory, so the inner wtr loop collapses to a single tile). Chain
         * emits the entry-time srca/srcb + pack reconfig that replaces the inline
         * reconfig_data_format / pack_reconfig_data_format / mul_tiles_init triad.
         * Pack uses FirstTile because PerTileReserveAndPush advances the write
         * pointer each push, making absolute idx wt equivalent to relative 0.
         */
        compute_kernel_lib::eltwise_chain<compute_kernel_lib::DEST_AUTO_LIMIT>(
            Wt,
            compute_kernel_lib::BinaryFpu<
                cb_inp,
                cb_inp,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::CopyTilePolicy::CumulativeWaitPopAtEnd,
                compute_kernel_lib::CopyTilePolicy::CumulativeWaitPopAtEnd,
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
        // BulkWaitBulkPop: All Wt tiles already in CB (chain pushed Wt tiles above)
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_x2, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        // cb_inp already popped by the chain (CumulativeWaitPopAtEnd) - no explicit pop here.
    }
    cb_pop_front(cb_reduce, 1);
}
