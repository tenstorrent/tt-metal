// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"  // dummy_pack
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
    const auto& dfb_in_token,
    const auto& dfb_max_scaler_token,
    const auto& dfb_max_token,
    const auto& dfb_out_token,
    typename DFB0,
    typename DFB1,
    typename DFB2,
    typename DFB3>
ALWI void calc_numeric_stable(DFB0& dfb_in_obj, DFB1& /*dfb_max_scaler_obj*/, DFB2& dfb_max_obj, DFB3& dfb_out_obj) {
    // Use reduce_helpers for MAX reduce (REDUCE_ROW, PRELOADED mode)
    // Note: The library handles waiting for scaler tile internally
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        dfb_in_token,
        dfb_max_scaler_token,
        dfb_max_token,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(compute_kernel_lib::ReduceInputBlockShape::row(block_w));

    // calculate x-max(x)
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(block_w).block_size(subblock_w),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Sub,
            ckl::input(dfb_in_token, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(dfb_max_token, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{
            dfb_in_obj, dfb_max_obj},
        ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_out_token, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled)>{
            dfb_out_obj});
    dfb_out_obj.wait_front(block_w);
}

void kernel_main() {
    constexpr std::uint32_t block_h = get_arg(args::block_h);
    constexpr std::uint32_t block_w = get_arg(args::block_w);
    constexpr std::uint32_t subblock_w = get_arg(args::subblock_w);
    constexpr bool causal_mask = get_arg(args::causal_mask) != 0;
    constexpr bool sharded_causal_mask = get_arg(args::sharded_causal_mask) != 0;
#ifdef NUMERIC_STABLE
    constexpr bool numeric_stable = true;
#else
    constexpr bool numeric_stable = false;
#endif

#ifdef NUMERIC_STABLE
    static constexpr auto dfb_x_id = dfb::x;
#else
    // Without numeric_stable, dfb_x aliases dfb_exps (Same-FIFO reuse).
    static constexpr auto dfb_x_id = dfb::exps;
#endif

    compute_kernel_hw_startup(dfb::in0, dfb::max_scaler, dfb::exps);

    DataflowBuffer dfb_exps_obj(dfb::exps);
    DataflowBuffer dfb_x_obj(dfb_x_id);
    DataflowBuffer dfb_max_scaler_obj(dfb::max_scaler);
    DataflowBuffer dfb_sum_scaler_obj(dfb::sum_scaler);
    // in0 and out0 are the borrowed input / output shards and this kernel is the only producer and consumer of
    // each (self-loop). The rows below pop in0 (which also advances to the next row) and push out0, so push
    // each in0 row before it is read and pop each out0 row after it is written to leave both balanced.
    DataflowBuffer dfb_in0_obj(dfb::in0);
    DataflowBuffer dfb_recip_sum_exps_obj(dfb::recip_sum_exps);
    DataflowBuffer dfb_out0_obj(dfb::out0);
#ifdef NUMERIC_STABLE
    DataflowBuffer dfb_max_obj(dfb::max);
#endif

#ifdef MASK_PADDED_DATA
    // The last dimension is not a multiple of the tile width: the reader pushes one -inf padding tile that the
    // last tile of every row gets added before the exp.
    DataflowBuffer dfb_mask_padded_obj(dfb::mask_padded);
    constexpr std::uint32_t block_w_unpadded = block_w - 1;
#else
    [[maybe_unused]] constexpr std::uint32_t block_w_unpadded = block_w;
#endif

#ifdef FUSED_SCALE_MASK
    DataflowBuffer dfb_fused_scale_obj(dfb::fused_scale);
    DataflowBuffer dfb_fused_attn_obj(dfb::fused_attn);
    DataflowBuffer dfb_scale_mask_obj(dfb::scale_mask);
    constexpr auto mask_bcast = causal_mask ? ckl::BroadcastDim::None : ckl::BroadcastDim::Row;
    constexpr auto mask_wait = sharded_causal_mask ? ckl::WaitPolicy::None : ckl::WaitPolicy::Upfront;
    constexpr auto mask_pop = causal_mask ? ckl::PopPolicy::AtEnd : ckl::PopPolicy::None;
#endif

    for (std::uint32_t i = 0; i < block_h; i++) {
        // The in0 row is already resident in L1; advance the DFB to make it available. On Quasar a push with
        // no intervening pack trips the TEN-4746 guard, so emit a no-write PACR first (a no-op on WH/BH).
        dfb_in0_obj.reserve_back(block_w);
        dummy_pack(dfb::in0);
        dfb_in0_obj.push_back(block_w);
#ifdef FUSED_SCALE_MASK
        if constexpr (block_w_unpadded > 0) {
            ckl::mul<
                ckl::input(dfb::in0, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::input(dfb::fused_scale, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
                ckl::output(dfb::scale_mask, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
                ckl::IterationShape::tiles(block_w_unpadded).block_size(subblock_w),
                dfb_in0_obj,
                dfb_fused_scale_obj,
                dfb_scale_mask_obj);
        }
#ifdef MASK_PADDED_DATA
        // The user mask does not cover the tile padding, so the last tile also gets the -inf padding mask.
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb::in0, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd),
                ckl::input(
                    dfb::fused_scale, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                dfb_in0_obj, dfb_fused_scale_obj},
            ckl::DestReuseBinary<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb::mask_padded, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
                ckl::DestReuseType::DEST_TO_SRCA>{dfb_mask_padded_obj},
            ckl::PackTile<ckl::output(dfb::scale_mask, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{
                dfb_scale_mask_obj});
#endif

        ckl::eltwise_chain(
            ckl::IterationShape::tiles(block_w).block_size(subblock_w),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(
                    dfb::scale_mask, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::input(dfb::fused_attn, mask_bcast, mask_wait, mask_pop, ckl::InputTileMapping::Block)>{
                dfb_scale_mask_obj, dfb_fused_attn_obj},
            // Exp dropped when NUMERIC_STABLE (it is fused into calc_numeric_stable below).
            ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(
                dfb_x_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled)>{
                dfb_x_obj});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        dfb_x_obj.wait_front(block_w);
        calc_numeric_stable<block_w, subblock_w, dfb_x_id, dfb::max_scaler, dfb::max, dfb::exps>(
            dfb_x_obj, dfb_max_scaler_obj, dfb_max_obj, dfb_exps_obj);
#endif

        reconfig_data_format(dfb::exps, dfb::sum_scaler);

#else

#if defined(MASK_PADDED_DATA) && defined(NUMERIC_STABLE) && !defined(ARCH_QUASAR)
        // Only the last tile of the row holds padding. Mask that tile into x and read the other tiles straight
        // from in0, rather than first copying the whole row into x (one extra pass over every tile of every row).
        // Quasar keeps the copy below: it has no MAX + REDUCE_ROW accumulation.
        if constexpr (block_w_unpadded > 0) {
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                dfb::in0,
                dfb::max_scaler,
                dfb::max,
                compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
                dfb_in0_obj,
                dfb_max_scaler_obj,
                dfb_max_obj,
                compute_kernel_lib::ReduceInputBlockShape::row(block_w_unpadded));
        }
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(
                    dfb::in0,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileAddressing::Offset),
                ckl::input(dfb::mask_padded, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                dfb_in0_obj, dfb_mask_padded_obj, block_w_unpadded},
            ckl::PackTile<ckl::output(dfb::x, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{dfb_x_obj});
        dfb_x_obj.wait_front(1);
        if constexpr (block_w_unpadded > 0) {
            // Fold the masked tile into the max of the other tiles (accumulator and output are both dfb::max).
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                dfb::x,
                dfb::max_scaler,
                dfb::max,
                compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
                dfb_x_obj,
                dfb_max_scaler_obj,
                dfb_max_obj,
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate(dfb_max_obj, /*iter=*/1));
        } else {
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                dfb::x,
                dfb::max_scaler,
                dfb::max,
                compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
                dfb_x_obj, dfb_max_scaler_obj, dfb_max_obj, compute_kernel_lib::ReduceInputBlockShape::row(1));
        }
        // exp(x - max): the unpadded tiles from in0 (keeping the max), then the masked tile from x.
        if constexpr (block_w_unpadded > 0) {
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(block_w_unpadded).block_size(subblock_w),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Sub,
                    ckl::input(dfb::in0, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                    ckl::input(dfb::max, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                    dfb_in0_obj, dfb_max_obj},
                ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(dfb::exps, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{
                    dfb_exps_obj});
        }
        // The raw last tile was read through x; release it from in0.
        dfb_in0_obj.pop_front(1);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Sub,
                ckl::input(dfb::x, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd),
                ckl::input(dfb::max, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{
                dfb_x_obj, dfb_max_obj},
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb::exps, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{dfb_exps_obj});
        dfb_exps_obj.wait_front(block_w);
#elif defined(MASK_PADDED_DATA)
        // Copy the row so the last tile can get the -inf padding mask before the max and the exp.
        if constexpr (block_w_unpadded > 0) {
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(block_w_unpadded).block_size(subblock_w),
                ckl::CopyTile<
                    ckl::input(dfb::in0, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                    ckl::Dst::D0>{dfb_in0_obj},
                ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(dfb_x_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{dfb_x_obj});
        }
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb::in0, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd),
                ckl::input(dfb::mask_padded, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                dfb_in0_obj, dfb_mask_padded_obj},
            ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(dfb_x_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{dfb_x_obj});
#ifdef NUMERIC_STABLE
        dfb_x_obj.wait_front(block_w);
        calc_numeric_stable<block_w, subblock_w, dfb_x_id, dfb::max_scaler, dfb::max, dfb::exps>(
            dfb_x_obj, dfb_max_scaler_obj, dfb_max_obj, dfb_exps_obj);
#endif
#elif defined(NUMERIC_STABLE)
        calc_numeric_stable<block_w, subblock_w, dfb::in0, dfb::max_scaler, dfb::max, dfb::exps>(
            dfb_in0_obj, dfb_max_scaler_obj, dfb_max_obj, dfb_exps_obj);
#else
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(block_w).block_size(subblock_w),
            ckl::CopyTile<
                ckl::input(dfb::in0, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
                ckl::Dst::D0>{dfb_in0_obj},
            ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb::exps, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{dfb_exps_obj});
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
            ckl::IterationShape::tiles(block_w).block_size(subblock_w),
            dfb_exps_obj,
            dfb_recip_sum_exps_obj,
            dfb_out0_obj);
        // The out0 row stays in place in the output shard; release it. dummy_unpack orders the pop after the
        // wait on Quasar (nothing unpacks out0).
        dfb_out0_obj.wait_front(block_w);
        dummy_unpack(dfb::out0);
        dfb_out0_obj.pop_front(block_w);
    }
#ifdef FUSED_SCALE_MASK
    // The fused-scale scalar is a single tile pushed once by the reader and re-waited on every row
    // of the block; pop it once here so the buffer is left balanced.
    dfb_fused_scale_obj.pop_front(1);
    // A non-causal attention mask is one row of block_w tiles that every row of the block re-waits
    // and reuses, so it is popped once here rather than per row. The causal paths set mask_pop to
    // AtEnd and so pop it inside the loop instead, so don't pop here in that case.
    if constexpr (!causal_mask) {
        dfb_fused_attn_obj.pop_front(block_w);
    }
#endif  // FUSED_SCALE_MASK
    // compute_kernel_lib::reduce waits the buffer it is given as the scaler and never pops it, so
    // one pushed tile serves every reduce call. Pop each scaler here, under the same condition that
    // gated its reductions, so the buffers are left balanced.
#ifdef NUMERIC_STABLE
    dfb_max_scaler_obj.pop_front(1);
#endif
    dfb_sum_scaler_obj.pop_front(1);
#ifdef MASK_PADDED_DATA
    // The padding mask is a single tile re-waited on the last tile of every row; pop it once here.
    dfb_mask_padded_obj.pop_front(1);
#endif
}
