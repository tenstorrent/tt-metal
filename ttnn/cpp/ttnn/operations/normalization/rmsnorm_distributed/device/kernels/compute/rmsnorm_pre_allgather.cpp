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
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/operations/normalization/kernel_util/compute/pre_add.h"
#include "experimental/kernel_args.h"

namespace pre_add = norm::kernel_util::compute::pre_add;

// The statistics pass reads either the fused a + b result or the raw input. fused is present
// iff residual was supplied; in0 is always real.
constexpr auto dfb_inp_id = engaged_token_between(dfb::fused, dfb::in0);

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::blk);
    constexpr bool unpack_fp32_active = get_arg(args::unpack_fp32_active) != 0;
    // Accurate mode only supports SUM; with the reader's scaler of 1.0, SUM and AVG are equivalent.
    constexpr auto reduce_type = unpack_fp32_active ? PoolType::SUM : PoolType::AVG;
    constexpr auto reduce_fp32_mode = unpack_fp32_active ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;

    constexpr uint32_t onetile = 1;

    with_nullable_token(
        dfb::res,
        [&](DFBBindingToken const& res) { compute_kernel_hw_startup(dfb::in0, res, dfb_inp_id); },
        [&] { compute_kernel_hw_startup(dfb_inp_id, dfb::reduce, dfb::x2); });

    DataflowBuffer dfb_inp(dfb_inp_id);
    DataflowBuffer dfb_x2(dfb::x2);
    DataflowBuffer dfb_reduce(dfb::reduce);
    DataflowBuffer dfb_in0(dfb::in0);
    auto dfb_res = construct_nullable_dfb(dfb::res);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Fuse pre-add: dfb_inp = dfb::in0 + dfb::res (absent entirely when there is no residual)
        with_nullable_dfb(dfb_res, [&](DataflowBuffer& dfb_res) {
            pre_add::one_row<true, unpack_fp32_active>(dfb_in0, dfb_res, dfb_inp, Wt, blk);
        });

        /*
         * x**2
         */
        reconfig_data_format(dfb_inp_id, dfb_inp_id);
        pack_reconfig_data_format(dfb::x2);
        if constexpr (unpack_fp32_active) {
            copy_tile_to_dst_init_short(dfb_inp_id);
            square_tile_init();
        } else {
            mul_init(dfb_inp_id, dfb_inp_id);
        }
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            dfb_inp.wait_front(wt + blk);  // cumulative wait
            dfb_x2.reserve_back(blk);

            if constexpr (unpack_fp32_active) {
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    tile_regs_acquire();
                    copy_tile(dfb_inp_id, wt + wtr, 0);
                    square_tile(0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, dfb::x2, wt + wtr);
                    tile_regs_release();
                }
            } else {
                tile_regs_acquire();
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    mul_tiles(dfb_inp_id, dfb_inp_id, wt + wtr, wt + wtr, wtr);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t wtr = 0; wtr < blk; wtr++) {
                    pack_tile(wtr, dfb::x2, wt + wtr);
                }
                tile_regs_release();
            }
            dfb_x2.push_back(blk);
        }

        /*
         * sum(x**2)
         */
        // BulkWaitBulkPop: All Wt tiles already in the buffer (see cumulative wait above)
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
    }
    dfb_reduce.pop_front(1);
}
