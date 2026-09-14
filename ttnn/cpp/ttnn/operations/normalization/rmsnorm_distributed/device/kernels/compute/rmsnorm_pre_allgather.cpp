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
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/operations/normalization/kernel_util/compute/pre_add.h"
#include "ttnn/operations/normalization/kernel_util/compute/pre_allgather_stats.h"
#include "experimental/kernel_args.h"

namespace pre_add = norm::kernel_util::compute::pre_add;
namespace pre_allgather = norm::kernel_util::compute::pre_allgather;

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
    constexpr bool unpack_fp32_active = get_arg(args::unpack_fp32_active) != 0;
    // Accurate mode only supports SUM; with the reader's scaler of 1.0, SUM and AVG are equivalent.
    constexpr auto reduce_type = unpack_fp32_active ? PoolType::SUM : PoolType::AVG;
    constexpr auto reduce_fp32_mode = unpack_fp32_active ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb::in0, dfb::res, dfb_inp_id);
#else
    compute_kernel_hw_startup(dfb_inp_id, dfb::reduce, dfb::x2);
#endif

    DataflowBuffer dfb_inp(dfb_inp_id);
    DataflowBuffer dfb_x2(dfb::x2);
    DataflowBuffer dfb_reduce(dfb::reduce);
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_res(dfb::res);  // residual b
#endif
#ifdef CHUNKED
    // The buffers hold chunk_wt tile-columns rather than the whole row; each chunk's sum(x**2) is
    // carried in dfb::acc_x2 and the reduce's Accumulate reloads it, so L1 stays bounded by chunk_wt.
    constexpr auto chunk_wt = get_arg(args::chunk_wt);
    constexpr std::uint32_t num_chunks = (Wt + chunk_wt - 1) / chunk_wt;
    DataflowBuffer dfb_acc_x2(dfb::acc_x2);
    DataflowBuffer dfb_out(dfb::out);
#endif

    for (std::uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef CHUNKED
        for (std::uint32_t c = 0; c < num_chunks; c++) {
            const std::uint32_t cw = (c == num_chunks - 1) ? Wt - c * chunk_wt : chunk_wt;
#ifdef FUSE_PRE_ADD
            pre_add::one_row<true, unpack_fp32_active>(dfb_in0, dfb_res, dfb_inp, cw, blk);
#endif
            pre_allgather::square_row<unpack_fp32_active>(dfb_inp, dfb_x2, cw, blk);
            dfb_inp.pop_front(cw);

            if constexpr (unpack_fp32_active) {
                // SFPU path: emit an independent partial per chunk; the epilogue folds them
                // (the reduce's Accumulate reload over-counts SUM partials, see fold_partials_to_out).
                compute_kernel_lib::reduce<
                    reduce_type,
                    ReduceDim::REDUCE_ROW,
                    dfb::x2,
                    dfb::reduce,
                    dfb::acc_x2,
                    compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    reduce_fp32_mode>(compute_kernel_lib::ReduceInputBlockShape::row(cw));
            } else {
                compute_kernel_lib::reduce<
                    reduce_type,
                    ReduceDim::REDUCE_ROW,
                    dfb::x2,
                    dfb::reduce,
                    dfb::acc_x2,
                    compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    reduce_fp32_mode>(
                    compute_kernel_lib::ReduceInputBlockShape::row(cw),
                    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                    compute_kernel_lib::Accumulate::at(dfb::acc_x2, c));
            }
        }
        if constexpr (unpack_fp32_active) {
            pre_allgather::fold_partials_to_out(dfb_acc_x2, dfb_out, num_chunks);
        } else {
            pre_allgather::partial_to_out(dfb_acc_x2, dfb_out);
        }
#else
#ifdef FUSE_PRE_ADD
        pre_add::one_row<true, unpack_fp32_active>(dfb_in0, dfb_res, dfb_inp, Wt, blk);
#endif
        pre_allgather::square_row<unpack_fp32_active>(dfb_inp, dfb_x2, Wt, blk);

        /*
         * sum(x**2)
         */
        // BulkWaitBulkPop: All Wt tiles already in the buffer (square_row's cumulative wait)
        compute_kernel_lib::reduce<
            reduce_type,
            ReduceDim::REDUCE_ROW,
            dfb::x2,
            dfb::reduce,
            dfb::out,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            reduce_fp32_mode>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        dfb_inp.pop_front(Wt);
#endif
    }
    dfb_reduce.pop_front(1);
}
