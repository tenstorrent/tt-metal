// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes distributed rmsnorm statistics: E(x**2).
 */

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // add

namespace ckl = compute_kernel_lib;

#ifdef FUSE_PRE_ADD
constexpr auto dfb_inp_id = dfb::fused;
#else
constexpr auto dfb_inp_id = dfb::in0;
#endif

void kernel_main() {
    constexpr auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto blk = get_arg(args::blk);
    constexpr auto num_cores_y = get_arg(args::num_cores_y);

#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb::in0, dfb::res, dfb_inp_id);
#else
    compute_kernel_hw_startup(dfb_inp_id, dfb::reduce, dfb::x2);
#endif

    constexpr auto squaring_shape = ckl::EltwiseShape::tiles(Wt, blk);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef FUSE_PRE_ADD
        ckl::add<
            ckl::input(dfb::in0, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::input(dfb::res, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::output(dfb_inp_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize),
            ckl::BroadcastDim::None>(squaring_shape);
#endif

        ckl::square<
            ckl::input(dfb_inp_id, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::OperandKind::Block),
            ckl::output(dfb::x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(squaring_shape);

        ckl::reduce<
            PoolType::AVG,
            ReduceDim::REDUCE_ROW,
            dfb::x2,
            dfb::reduce,
            dfb::out,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(ckl::ReduceInputBlockShape::row(Wt));
        DataflowBuffer(dfb_inp_id).pop_front(Wt);
        DataflowBuffer(dfb::reduce).pop_front(1);
    }

#ifdef IS_MERGE_CORE
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_cores_y),
        ckl::BinaryFpu<
            ckl::input(dfb::x2_merge, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(dfb::zero, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None,
            ckl::Dst::D0,
            ckl::DestAccumulation::WholeShape>{},
        ckl::PackTile<ckl::output(
            dfb::out_final,
            ckl::ReservePolicy::PerOuter,
            ckl::PushPolicy::PerOuter,
            ckl::DataFormatReconfig::Enabled,
            ckl::PackRelu::Disabled,
            ckl::L1Accumulation::Disabled,
            ckl::DestAccumulation::WholeShape)>{});
#endif
}
