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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

template <
    uint32_t block_w,
    uint32_t num_subblocks_w,
    uint32_t subblock_w,
    uint32_t dfb_in_id,
    uint32_t dfb_max_scaler_id,
    uint32_t dfb_max_id,
    uint32_t dfb_out_id>
ALWI void calc_numeric_stable() {
    auto dfb_out_obj_id = DataflowBuffer(dfb_out_id);

    // Use reduce_helpers for MAX reduce (REDUCE_ROW, PRELOADED mode)
    // Note: The library handles waiting for scaler tile internally
    ckl::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        dfb_in_id,
        dfb_max_scaler_id,
        dfb_max_id,
        ckl::ReduceInputPolicy::NoWaitNoPop>(ckl::ReduceInputBlockShape::row(block_w));

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(block_w, subblock_w),
        ckl::BinaryFpu<
            ckl::input(dfb_in_id, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(dfb_max_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::BinaryFpuOp::Sub,
            ckl::BroadcastDim::Col>{},
        ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_out_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled)>{});
    dfb_out_obj_id.wait_front(block_w);
}

void kernel_main() {
    constexpr std::uint32_t block_h = get_arg(args::block_h);
    constexpr std::uint32_t block_w = get_arg(args::block_w);
    constexpr std::uint32_t subblock_w = get_arg(args::subblock_w);
    constexpr std::uint32_t num_subblocks_w = get_arg(args::num_subblocks_w);

    constexpr auto dfb_in0_id = dfb::in0;
    constexpr auto dfb_max_scaler_id = dfb::max_scaler;
    constexpr auto dfb_sum_scaler_id = dfb::sum_scaler;
    constexpr auto dfb_exps_id = dfb::exps;
    constexpr auto dfb_recipsumexps_id = dfb::recip_sum_exps;
    constexpr auto dfb_out0_id = dfb::out0;
#if FUSED_SCALE_MASK
    constexpr auto dfb_fused_scale_id = dfb::fused_scale;
    constexpr auto dfb_fused_attn_id = dfb::fused_attn;
    constexpr auto dfb_scale_mask_id = dfb::scale_mask;
#endif
#ifdef NUMERIC_STABLE
    constexpr auto dfb_max_id = dfb::max;
    constexpr auto dfb_x_id = dfb::x;
#else
    constexpr auto dfb_x_id = dfb_exps_id;
#endif

    compute_kernel_hw_startup(dfb_in0_id, dfb_max_scaler_id, dfb_exps_id);

    auto dfb_in0_obj_id = DataflowBuffer(dfb_in0_id);
    auto dfb_max_scaler_obj_id = DataflowBuffer(dfb_max_scaler_id);
    auto dfb_exps_obj_id = DataflowBuffer(dfb_exps_id);
    auto dfb_out0_obj_id = DataflowBuffer(dfb_out0_id);
    auto dfb_x_obj_id = DataflowBuffer(dfb_x_id);
#if FUSED_SCALE_MASK
    auto dfb_fused_attn_obj_id = DataflowBuffer(dfb_fused_attn_id);
    auto dfb_scale_mask_obj_id = DataflowBuffer(dfb_scale_mask_id);
#endif
#ifdef NUMERIC_STABLE
    auto dfb_max_obj_id = DataflowBuffer(dfb_max_id);
#endif

    constexpr int dst0 = 0;

#if FUSED_SCALE_MASK
#ifdef CAUSAL_MASK
    constexpr bool causal_mask = true;
#else
    constexpr bool causal_mask = false;
#endif
#ifdef SHARDED_CAUSAL_MASK
    constexpr bool sharded_causal_mask = true;
#else
    constexpr bool sharded_causal_mask = false;
#endif
#ifdef NUMERIC_STABLE
    constexpr bool numeric_stable = true;
#else
    constexpr bool numeric_stable = false;
#endif
    constexpr auto mask_bcast = causal_mask ? ckl::BroadcastDim::None : ckl::BroadcastDim::Row;
    constexpr auto mask_wait = sharded_causal_mask ? ckl::WaitPolicy::None : ckl::WaitPolicy::Upfront;
    constexpr auto mask_pop = causal_mask ? ckl::PopPolicy::AtEnd : ckl::PopPolicy::None;
#endif

    for (std::uint32_t i = 0; i < block_h; i++) {
#if FUSED_SCALE_MASK
        ckl::mul<
            ckl::input(dfb_in0_id, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(dfb_fused_scale_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
            ckl::output(dfb_scale_mask_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd),
            ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::tiles(block_w, subblock_w));

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(block_w, subblock_w),
            ckl::BinaryFpu<
                ckl::input(dfb_scale_mask_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(dfb_fused_attn_id, mask_wait, mask_pop, ckl::OperandKind::Block),
                ckl::BinaryFpuOp::Add,
                mask_bcast>{},
            // Exp dropped when NUMERIC_STABLE (it is fused into calc_numeric_stable below).
            ckl::OptionalChainElement<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(
                dfb_x_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        dfb_x_obj_id.wait_front(block_w);
        calc_numeric_stable<
            block_w,
            num_subblocks_w,
            subblock_w,
            dfb_x_id,
            dfb_max_scaler_id,
            dfb_max_id,
            dfb_exps_id>();
#endif

        reconfig_data_format(dfb_exps_id, dfb_sum_scaler_id);

#else

#ifdef NUMERIC_STABLE
        calc_numeric_stable<
            block_w,
            num_subblocks_w,
            subblock_w,
            dfb_in0_id,
            dfb_max_scaler_id,
            dfb_max_id,
            dfb_exps_id>();
#else
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(block_w, subblock_w),
            ckl::CopyTile<
                ckl::input(dfb_in0_id, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::Dst::D0>{},
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb_exps_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{});
#endif
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal operation using PRELOADED mode
        // PRELOADED is correct for sharded - all tiles loaded at once
        // Auto-detects FP32 mode from ENABLE_FP32_DEST_ACC define
        DataflowBuffer(dfb_exps_id).wait_front(block_w);
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb_exps_id,
            dfb_sum_scaler_id,
            dfb_recipsumexps_id,
            ckl::ReduceInputPolicy::NoWaitNoPop>(
            ckl::ReduceInputBlockShape::row(block_w),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t) {
                recip_tile_init();
                recip_tile(0);
            });

        ckl::mul<
            ckl::input(dfb_exps_id, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(dfb_recipsumexps_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb_out0_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(block_w, subblock_w));
    }
}
