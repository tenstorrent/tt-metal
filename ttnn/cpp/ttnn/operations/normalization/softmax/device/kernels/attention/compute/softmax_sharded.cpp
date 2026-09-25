// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

template <
    std::uint32_t block_w,
    std::uint32_t subblock_w,
    std::uint32_t dfb_in_id,
    std::uint32_t dfb_max_scaler_id,
    std::uint32_t dfb_max_id,
    std::uint32_t dfb_out_id>
ALWI void calc_numeric_stable() {
    DataflowBuffer dfb_out_obj(dfb_out_id);

    // Use reduce_helpers for MAX reduce (REDUCE_ROW, PRELOADED mode)
    // Note: The library handles waiting for scaler tile internally
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        dfb_in_id,
        dfb_max_scaler_id,
        dfb_max_id,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(compute_kernel_lib::ReduceInputBlockShape::row(block_w));

    // calculate x-max(x)
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(block_w).block_size(subblock_w),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Sub,
            ckl::input(dfb_in_id, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(dfb_max_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{},
        ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_out_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled)>{});
    dfb_out_obj.wait_front(block_w);
}

void kernel_main() {
    constexpr std::uint32_t block_h = get_arg(args::block_h);
    constexpr std::uint32_t block_w = get_arg(args::block_w);
    constexpr std::uint32_t subblock_w = get_arg(args::subblock_w);
    constexpr bool causal_mask = get_arg(args::causal_mask);
    constexpr bool sharded_causal_mask = get_arg(args::sharded_causal_mask);
#ifdef NUMERIC_STABLE
    constexpr bool numeric_stable = true;
#else
    constexpr bool numeric_stable = false;
#endif

#ifdef NUMERIC_STABLE
    constexpr auto dfb_x_id = dfb::x;
#else
    // Without numeric_stable, dfb_x aliases dfb_exps (Same-FIFO reuse).
    constexpr auto dfb_x_id = dfb::exps;
#endif

    compute_kernel_hw_startup(dfb::in0, dfb::max_scaler, dfb::exps);

    DataflowBuffer dfb_exps_obj(dfb::exps);
    DataflowBuffer dfb_x_obj(dfb_x_id);

#ifdef FUSED_SCALE_MASK
    constexpr auto mask_bcast = causal_mask ? ckl::BroadcastDim::None : ckl::BroadcastDim::Row;
    constexpr auto mask_wait = sharded_causal_mask ? ckl::WaitPolicy::None : ckl::WaitPolicy::Upfront;
    constexpr auto mask_pop = causal_mask ? ckl::PopPolicy::AtEnd : ckl::PopPolicy::None;
#endif

    for (std::uint32_t i = 0; i < block_h; i++) {
#ifdef FUSED_SCALE_MASK
        ckl::mul<
            ckl::input(dfb::in0, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(dfb::fused_scale, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
            ckl::output(dfb::scale_mask, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
            ckl::IterationShape::tiles(block_w).block_size(subblock_w));

        ckl::eltwise_chain(
            ckl::IterationShape::tiles(block_w).block_size(subblock_w),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(
                    dfb::scale_mask, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::input(dfb::fused_attn, mask_bcast, mask_wait, mask_pop, ckl::InputTileMapping::Block)>{},
            // Exp dropped when NUMERIC_STABLE (it is fused into calc_numeric_stable below).
            ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(
                dfb_x_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        dfb_x_obj.wait_front(block_w);
        calc_numeric_stable<block_w, subblock_w, dfb_x_id, dfb::max_scaler, dfb::max, dfb::exps>();
#endif

        reconfig_data_format(dfb::exps, dfb::sum_scaler);

#else

#ifdef NUMERIC_STABLE
        calc_numeric_stable<block_w, subblock_w, dfb::in0, dfb::max_scaler, dfb::max, dfb::exps>();
#else
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(block_w).block_size(subblock_w),
            ckl::CopyTile<
                ckl::input(dfb::in0, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::Dst::D0>{},
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb::exps, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{});
#endif
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal operation using PRELOADED mode
        // PRELOADED is correct for sharded - all tiles loaded at once
        // Auto-detects FP32 mode from ENABLE_FP32_DEST_ACC define
        dfb_exps_obj.wait_front(block_w);
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb::exps,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
            compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(block_w),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](std::uint32_t) {
                // Preserve the FP32 row sum's precision in its reciprocal, independently of exp approximation.
                if constexpr (DST_ACCUM_MODE) {
                    recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
                    recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0);
                } else {
                    recip_tile_init();
                    recip_tile(0);
                }
            });

        ckl::mul<
            ckl::input(dfb::exps, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(dfb::recip_sum_exps, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb::out0, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
            ckl::IterationShape::tiles(block_w).block_size(subblock_w));
    }
}
