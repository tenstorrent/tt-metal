// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
 * For rmsnorm we compute E(x**2) and return it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 *
 * Metal 2.0 fork of rmsnorm_pre_allgather.cpp: same computation, with named kernel arguments and
 * named dataflow-buffer bindings instead of positional compile-time args and CB indices. The legacy
 * file beside this one still serves consumers that have not migrated.
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/operations/normalization/kernel_util/compute/pre_add.h"
#include "experimental/kernel_args.h"

namespace pre_add = norm::kernel_util::compute::pre_add;

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

// The statistics pass reads either the raw input or the fused a + b result, depending on whether a
// residual was supplied. Only the buffer selected here is bound on this build, so the alias is gated
// at the preprocessor: naming an unbound handle would not compile even on a discarded branch.
#ifdef FUSE_PRE_ADD
constexpr auto dfb_inp_id = dfb::fused;  // fused a + b
#else
constexpr auto dfb_inp_id = dfb::in0;  // just a
#endif

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::blk);

    constexpr uint32_t onetile = 1;

#ifdef FUSE_PRE_ADD
    binary_op_init_common(dfb::in0, dfb::res, dfb_inp_id);
#else
    binary_op_init_common(dfb_inp_id, dfb::reduce, dfb::x2);
#endif

    DataflowBuffer dfb_inp(dfb_inp_id);
    DataflowBuffer dfb_x2(dfb::x2);
    DataflowBuffer dfb_reduce(dfb::reduce);
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_res(dfb::res);  // residual b
#endif

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: dfb_inp = dfb::in0 + dfb::res (absent entirely when there is no residual)
#ifdef FUSE_PRE_ADD
        pre_add::one_row<true>(dfb_in0, dfb_res, dfb_inp, Wt, blk);
#endif

        /*
         * x**2
         */
        reconfig_data_format(dfb_inp_id, dfb_inp_id);
        pack_reconfig_data_format(dfb::x2);
        mul_tiles_init(dfb_inp_id, dfb_inp_id);
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            dfb_inp.wait_front(wt + blk);  // cumulative wait
            dfb_x2.reserve_back(blk);
            ACQ();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(dfb_inp_id, dfb_inp_id, wt + wtr, wt + wtr, wtr);
                pack_tile(wtr, dfb::x2, wt + wtr);
            }
            REL();
            dfb_x2.push_back(blk);
        }

        /*
         * sum(x**2)
         */
        // BulkWaitBulkPop: All Wt tiles already in the buffer (see cumulative wait above)
        compute_kernel_lib::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            dfb::x2,
            dfb::reduce,
            dfb::out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        dfb_inp.pop_front(Wt);
    }
    dfb_reduce.pop_front(1);
}
