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

// Templated on the CBs so the eltwise_chain below can take them as compile-time NTTPs.
template <
    uint32_t block_w,
    uint32_t num_subblocks_w,
    uint32_t subblock_w,
    uint32_t cb_in,
    uint32_t cb_max_scaler,
    uint32_t cb_max,
    uint32_t cb_out>
ALWI void calc_numeric_stable() {
    auto cb_max_obj = CircularBuffer(cb_max);
    auto cb_out_obj = CircularBuffer(cb_out);

    // Use reduce_helpers for MAX reduce (REDUCE_ROW, PRELOADED mode)
    // Note: The library handles waiting for scaler tile internally
    compute_kernel_lib::
        reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
            cb_in, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::row(block_w));

    // x - max(x) then exp, fused — DEST-batched subblock_w tiles per acquire, matching the
    // original's `for (j < num_subblocks_w) { for (w < subblock_w) }` window with absolute
    // index `w + index_subblock_w_offset`. cb_in resident (popped at chain end) -> DeferredPop
    // + Block, read by absolute index `wt_base + j`; cb_max held (wait/pop kept outside) ->
    // CallerManaged + Scalar; cb_out reserve+push block_w upfront/walk -> Bulk (block-correct).
    // EltwiseShape::tiles(block_w, subblock_w): subblock_w is the block_size; the chain clamps
    // it to DEST capacity automatically. sub_bcast_cols_init_short ->
    // BinaryDataFormatReconfig::Input; plain pack_tile (format already cb_out) ->
    // PackTileReconfig::None.
    cb_max_obj.wait_front(1);
    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(block_w, subblock_w),
        compute_kernel_lib::BinaryFpu<
            cb_in,
            cb_max,
            compute_kernel_lib::BinaryFpuOp::Sub,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::InputLifecycle::DeferredPop,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::OperandKind::Scalar>{},
        compute_kernel_lib::Exp<
            static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
            compute_kernel_lib::Approx::Exact,
            compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::
            PackTile<cb_out, compute_kernel_lib::OutputLifecycle::Bulk, compute_kernel_lib::PackTileReconfig::None>{});
    cb_max_obj.pop_front(1);
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
    auto cb_fused_scale_obj = CircularBuffer(cb_fused_scale);
    auto cb_fused_attn_obj = CircularBuffer(cb_fused_attn);
    auto cb_exps_obj = CircularBuffer(cb_exps);
    auto cb_recipsumexps_obj = CircularBuffer(cb_recipsumexps);
    auto cb_scale_mask_obj = CircularBuffer(cb_scale_mask);
    auto cb_out0_obj = CircularBuffer(cb_out0);
    auto cb_x_obj = CircularBuffer(cb_x);
#ifdef NUMERIC_STABLE
    auto cb_max_obj = CircularBuffer(cb_max);
#endif

    constexpr int dst0 = 0;

    for (uint32_t i = 0; i < block_h; i++) {
#if FUSED_SCALE_MASK
        // fused scale — DEST-batched subblock_w tiles per acquire, matching the original's
        // per-subblock window with absolute index `w + index_subblock_w_offset`. cb_in0
        // (resident sharded, popped at chain end) * cb_fused_scale (held scalar) ->
        // cb_scale_mask. cb_in0 DeferredPop + Block (absolute index `wt_base + j`);
        // cb_fused_scale CallerManaged + Scalar (wait kept below); cb_scale_mask reserve+push
        // block_w upfront/walk -> Bulk (block-correct). reconfig +
        // mul_tiles_bcast_scalar_init_short -> Input; pack_reconfig -> Output.
        // EltwiseShape::tiles(block_w, subblock_w): subblock_w is the block_size, DEST-clamped.
        cb_fused_scale_obj.wait_front(1);
        compute_kernel_lib::mul<
            cb_in0,
            cb_fused_scale,
            cb_scale_mask,
            compute_kernel_lib::BroadcastDim::Scalar,
            compute_kernel_lib::InputLifecycle::DeferredPop,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::OutputLifecycle::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::OperandKind::Scalar>(compute_kernel_lib::EltwiseShape::tiles(block_w, subblock_w));

        // fused mask add (+exp) — DEST-batched subblock_w tiles per acquire, matching the
        // original's per-subblock window with absolute index `w + index_subblock_w_offset`.
        // cb_scale_mask (Bulk: wait+pop block_w, absolute index `wt_base + j`) + cb_fused_attn
        // (lifecycle per the CAUSAL_MASK / SHARDED_CAUSAL_MASK matrix — chain owns wait+pop,
        // exactly reproducing the original's #ifndef SHARDED_CAUSAL_MASK wait + #ifdef CAUSAL_MASK
        // pop) -> cb_x (reserve+push block_w upfront/walk -> Bulk). add init -> Input; plain
        // pack_tile -> None. Exp dropped when NUMERIC_STABLE (done in calc_numeric_stable below).
        // EltwiseShape::tiles(block_w, subblock_w): subblock_w is the block_size, DEST-clamped.
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::EltwiseShape::tiles(block_w, subblock_w),
            compute_kernel_lib::BinaryFpu<
                cb_scale_mask,
                cb_fused_attn,
                compute_kernel_lib::BinaryFpuOp::Add,
#ifdef CAUSAL_MASK
                compute_kernel_lib::BroadcastDim::None,
#else
                compute_kernel_lib::BroadcastDim::Row,
#endif
                compute_kernel_lib::InputLifecycle::Bulk,
#if defined(CAUSAL_MASK) && defined(SHARDED_CAUSAL_MASK)
                compute_kernel_lib::InputLifecycle::DeferredPop,  // no wait (SHARDED), pop (CAUSAL)
#elif defined(CAUSAL_MASK)
                compute_kernel_lib::InputLifecycle::Bulk,  // wait + pop
#elif !defined(SHARDED_CAUSAL_MASK)
                compute_kernel_lib::InputLifecycle::HeldBulk,  // wait, no pop (held across block_h)
#else
                compute_kernel_lib::InputLifecycle::CallerManaged,  // no wait, no pop
#endif
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::OperandKind::Block>{},
#ifndef NUMERIC_STABLE
            compute_kernel_lib::Exp<
                static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
#endif
            compute_kernel_lib::PackTile<
                cb_x,
                compute_kernel_lib::OutputLifecycle::Bulk,
                compute_kernel_lib::PackTileReconfig::None>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        cb_x_obj.wait_front(block_w);
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w, cb_x, cb_max_scaler, cb_max, cb_exps>();
#endif

        // cb_fused_attn pop is now owned by the fused-attn chain above (DeferredPop/Bulk under
        // CAUSAL_MASK; held otherwise) — no external pop needed.
        reconfig_data_format(cb_exps, cb_sum_scaler);

#else

#ifdef NUMERIC_STABLE
        calc_numeric_stable<block_w, num_subblocks_w, subblock_w, cb_in0, cb_max_scaler, cb_max, cb_exps>();
#else
        // exp(x): CopyTile + Exp + PackTile, DEST-batched subblock_w tiles per acquire over the
        // whole block_w — matching the original's `for (j < num_subblocks_w) { for (w <
        // subblock_w) copy_tile(cb_in0, w + index_subblock_w_offset, w) }` window with absolute
        // index `w + index_subblock_w_offset`. cb_in0 is sharded-resident (block_w*block_h),
        // read by absolute index `wt_base + j` and popped block_w at chain end -> DeferredPop +
        // Block (the prior per-subblock loop with TileOffset::Set was BlockSize=1; this is the
        // single batched form). cb_exps reserve+push block_w upfront/walk -> Bulk (block-correct).
        // Reconfig: CopyTileReconfig::Input (== original reconfig_data_format(cb_in0, cb_in0) +
        // copy_tile_to_dst_init_short) + PackTileReconfig::Output (== pack_reconfig_data_format(
        // cb_exps)). EltwiseShape::tiles(block_w, subblock_w): subblock_w is the block_size,
        // DEST-clamped.
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::EltwiseShape::tiles(block_w, subblock_w),
            compute_kernel_lib::CopyTile<
                cb_in0,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::DeferredPop,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::OperandKind::Block>{},
            compute_kernel_lib::Exp<
                static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_exps,
                compute_kernel_lib::OutputLifecycle::Bulk,
                compute_kernel_lib::PackTileReconfig::Output>{});
#endif
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal operation using PRELOADED mode
        // PRELOADED is correct for sharded - all tiles loaded at once
        // Auto-detects FP32 mode from ENABLE_FP32_DEST_ACC define
        cb_wait_front(cb_exps, block_w);
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
                cb_exps,
                cb_sum_scaler,
                cb_recipsumexps,
                compute_kernel_lib::ReduceInputBlockShape::row(block_w),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t) {
                    recip_tile_init();
                    recip_tile(0);
                });

        // exp(x) / (sum(exp(x))) — bcast COL on cb_recipsumexps (1 tile held) — DEST-batched
        // subblock_w tiles per acquire, matching the original's per-subblock window with
        // absolute index `w + index_subblock_w_offset`. The SUM reduce above (NoWaitNoPop) ran
        // after cb_wait_front(cb_exps, block_w), so cb_exps is fully resident (block_w): it
        // walks by absolute index `wt_base + j` -> DeferredPop + Block, chain pops block_w at
        // end (== the original's lone cb_exps_obj.pop_front(block_w) after the mul loop).
        // cb_recipsumexps held outside (wait/pop bracket the chain) -> CallerManaged + Scalar.
        // cb_out0 (block_w*block_h resident) reserve+push block_w upfront/walk -> Bulk.
        //
        // Reconfig: reconfig_data_format + mul_bcast_cols_init_short -> Input;
        // pack_reconfig_data_format(cb_out0) -> PackTileReconfig::Output.
        // EltwiseShape::tiles(block_w, subblock_w): subblock_w is the block_size, DEST-clamped.
        cb_recipsumexps_obj.wait_front(1);
        compute_kernel_lib::mul<
            cb_exps,
            cb_recipsumexps,
            cb_out0,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::InputLifecycle::DeferredPop,
            compute_kernel_lib::InputLifecycle::CallerManaged,
            compute_kernel_lib::OutputLifecycle::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            compute_kernel_lib::OperandKind::Block,
            compute_kernel_lib::OperandKind::Scalar>(compute_kernel_lib::EltwiseShape::tiles(block_w, subblock_w));
        cb_recipsumexps_obj.pop_front(1);
    }
}
