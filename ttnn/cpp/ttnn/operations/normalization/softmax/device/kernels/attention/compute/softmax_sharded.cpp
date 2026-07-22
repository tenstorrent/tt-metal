// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

namespace ckl = compute_kernel_lib;

template <
    uint32_t block_w,
    uint32_t num_subblocks_w,
    uint32_t subblock_w,
    uint32_t cb_in,
    uint32_t cb_max_scaler,
    uint32_t cb_max,
    uint32_t cb_out>
ALWI void calc_numeric_stable() {
    auto cb_out_obj = CircularBuffer(cb_out);

    // Use reduce_helpers for MAX reduce (REDUCE_ROW, PRELOADED mode)
    // Note: The library handles waiting for scaler tile internally
    ckl::
        reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_in, cb_max_scaler, cb_max, ckl::ReduceInputPolicy::NoWaitNoPop>(
            ckl::ReduceInputBlockShape::row(block_w));

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(block_w, subblock_w),
        ckl::BinaryFpu<
            ckl::input(cb_in, ckl::InputLifecycle::DeferredPop, ckl::OperandKind::Block),
            ckl::input(cb_max, ckl::InputLifecycle::Bulk),
            ckl::BinaryFpuOp::Sub,
            ckl::BroadcastDim::Col>{},
        ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Approx::Exact, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(cb_out, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>{});
    cb_out_obj.wait_front(block_w);
}

void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(2);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(3);

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_6);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_max_scaler = tt::CBIndex::c_1;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_13;
    constexpr auto cb_fused_scale = tt::CBIndex::c_2;
    constexpr auto cb_fused_attn = tt::CBIndex::c_3;
    constexpr auto cb_exps = tt::CBIndex::c_6;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_7;
    constexpr auto cb_scale_mask = tt::CBIndex::c_8;
    constexpr auto cb_out0 = tt::CBIndex::c_11;
#ifdef NUMERIC_STABLE
    constexpr auto cb_max = tt::CBIndex::c_9;
    constexpr auto cb_x = tt::CBIndex::c_10;
#else
    constexpr auto cb_x = cb_exps;
#endif

    auto cb_in0_obj = CircularBuffer(cb_in0);
    auto cb_max_scaler_obj = CircularBuffer(cb_max_scaler);
    auto cb_fused_attn_obj = CircularBuffer(cb_fused_attn);
    auto cb_exps_obj = CircularBuffer(cb_exps);
    auto cb_scale_mask_obj = CircularBuffer(cb_scale_mask);
    auto cb_out0_obj = CircularBuffer(cb_out0);
    auto cb_x_obj = CircularBuffer(cb_x);
#ifdef NUMERIC_STABLE
    auto cb_max_obj = CircularBuffer(cb_max);
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
    constexpr auto mask_lifecycle = (causal_mask && sharded_causal_mask) ? ckl::InputLifecycle::DeferredPop
                                    : causal_mask                        ? ckl::InputLifecycle::Bulk
                                    : !sharded_causal_mask               ? ckl::InputLifecycle::HeldBulk
                                                                         : ckl::InputLifecycle::CallerManaged;
#endif

    for (uint32_t i = 0; i < block_h; i++) {
#if FUSED_SCALE_MASK
        ckl::mul<
            ckl::input(cb_in0, ckl::InputLifecycle::DeferredPop, ckl::OperandKind::Block),
            ckl::input(cb_fused_scale, ckl::InputLifecycle::HeldBulk),
            ckl::output(cb_scale_mask, ckl::OutputLifecycle::Bulk),
            ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::tiles(block_w, subblock_w));

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(block_w, subblock_w),
            ckl::BinaryFpu<
                ckl::input(cb_scale_mask, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                ckl::input(cb_fused_attn, mask_lifecycle, ckl::OperandKind::Block),
                ckl::BinaryFpuOp::Add,
                mask_bcast>{},
            // Exp dropped when NUMERIC_STABLE (it is fused into calc_numeric_stable below).
            ckl::OptionalChainElement<
                !numeric_stable,
                ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Approx::Exact, ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(cb_x, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        cb_x_obj.wait_front(block_w);
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w, cb_x, cb_max_scaler, cb_max, cb_exps>();
#endif

        reconfig_data_format(cb_exps, cb_sum_scaler);

#else

#ifdef NUMERIC_STABLE
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w, cb_in0, cb_max_scaler, cb_max, cb_exps>();
#else
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(block_w, subblock_w),
            ckl::
                CopyTile<ckl::input(cb_in0, ckl::InputLifecycle::DeferredPop, ckl::OperandKind::Block), ckl::Dst::D0>{},
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_exps, ckl::OutputLifecycle::Bulk)>{});
#endif
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal operation using PRELOADED mode
        // PRELOADED is correct for sharded - all tiles loaded at once
        // Auto-detects FP32 mode from ENABLE_FP32_DEST_ACC define
        cb_wait_front(cb_exps, block_w);
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            ckl::ReduceInputPolicy::NoWaitNoPop>(
            ckl::ReduceInputBlockShape::row(block_w),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t) {
                recip_tile_init();
                recip_tile(0);
            });

        ckl::mul<
            ckl::input(cb_exps, ckl::InputLifecycle::DeferredPop, ckl::OperandKind::Block),
            ckl::input(cb_recipsumexps, ckl::InputLifecycle::Bulk),
            ckl::output(cb_out0, ckl::OutputLifecycle::Bulk),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(block_w, subblock_w));
    }
}
