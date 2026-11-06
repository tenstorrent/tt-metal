// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const std::uint32_t T = get_arg(args::T);
    const std::uint32_t c = get_arg(args::c);
    const std::uint32_t C = get_arg(args::C);
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const std::uint32_t ndst = get_arg(args::blk);
    if (T == 0 || num_rows == 0) {
        return;
    }
    static constexpr auto dfb_in0 = dfb::in0;
    static constexpr auto dfb_max_scaler = dfb::max_scaler;
    static constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    static constexpr auto dfb_exps = dfb::exps;
    static constexpr auto dfb_pmax = dfb::pmax;
    static constexpr auto dfb_mcb = dfb::mcb;
    static constexpr auto dfb_out0 = dfb::out0;
    static constexpr auto dfb_rcb = dfb::rcb;
    DataflowBuffer in0(dfb_in0);
    DataflowBuffer max_scaler(dfb_max_scaler);
    DataflowBuffer sum_scaler(dfb_sum_scaler);
    DataflowBuffer exps(dfb_exps);
    DataflowBuffer pmax(dfb_pmax);
    DataflowBuffer rmax(dfb::rmax);
    DataflowBuffer mout(dfb::mout);
    DataflowBuffer mcb(dfb_mcb);
    DataflowBuffer pacc(dfb::pacc);
    DataflowBuffer racc(dfb::racc);
    DataflowBuffer rout(dfb::rout);
    DataflowBuffer rcb(dfb_rcb);
    DataflowBuffer out0(dfb_out0);

    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_exps);
    max_scaler.wait_front(1);
    sum_scaler.wait_front(1);
    const bool is_last = c + 1 == C;
    const std::uint32_t sum_out =
        is_last ? static_cast<std::uint32_t>(dfb::rout) : static_cast<std::uint32_t>(dfb::pacc);

    for (std::uint32_t r = 0; r < num_rows; ++r) {
        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_ROW,
            dfb_in0,
            dfb_max_scaler,
            dfb_pmax,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            in0, max_scaler, pmax, compute_kernel_lib::ReduceInputBlockShape::row(T));

        if (c == 0) {
            rmax.wait_front(static_cast<uint16_t>(C));
            reconfig_data_format_srca(dfb::rmax);
            pack_reconfig_data_format(dfb::mout);
            tile_regs_acquire();
            copy_init(dfb::rmax);
            copy_tile(dfb::rmax, 0, 0);
            for (std::uint32_t i = 1; i < C; ++i) {
                copy_init(dfb::rmax);
                copy_tile(dfb::rmax, i, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);
            }
            tile_regs_commit();
            mout.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, dfb::mout);
            tile_regs_release();
            mout.push_back(1);
            rmax.pop_front(static_cast<uint16_t>(C));
        }

        reconfig_data_format(dfb_in0, dfb_mcb);
        pack_reconfig_data_format(dfb_exps);
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(T).block_size(ndst),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Sub,
                ckl::input(
                    dfb_in0,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(dfb_mcb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{in0, mcb},
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_exps,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>{exps});
        exps.wait_front(static_cast<uint16_t>(T));

        reconfig_data_format(dfb_sum_scaler, dfb_exps);
        if (c > 0) {
            racc.wait_front(1);
        }
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_sum_scaler, sum_out);
        tile_regs_acquire();
        if (c > 0) {
            reconfig_data_format_srca(dfb::racc);
            copy_init(dfb::racc);
            copy_tile(dfb::racc, 0, 0);
            reconfig_data_format(dfb_sum_scaler, dfb_exps);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_sum_scaler, sum_out);
        }
        for (std::uint32_t t = 0; t < T; ++t) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_exps, dfb_sum_scaler, t, 0, 0);
        }
        if (is_last) {
            if constexpr (DST_ACCUM_MODE) {
                recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
                recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0);
            } else {
                recip_tile_init();
                recip_tile(0);
            }
            pack_reconfig_data_format(dfb::rout);
            rout.reserve_back(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::rout);
            tile_regs_release();
            rout.push_back(1);
            reduce_uninit();
        } else {
            pack_reconfig_data_format(dfb::pacc);
            pacc.reserve_back(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::pacc);
            tile_regs_release();
            pacc.push_back(1);
            reduce_uninit();
        }
        if (c > 0) {
            racc.pop_front(1);
        }

        ckl::mul<
            ckl::input(dfb_exps, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(dfb_rcb, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb_out0, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::tiles(T).block_size(ndst), exps, rcb, out0);
    }
}
