// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>

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

constexpr bool kCausalMask =
#ifdef CAUSAL_MASK
    true;
#else
    false;
#endif
constexpr bool kShardedCausalMask =
#ifdef SHARDED_CAUSAL_MASK
    true;
#else
    false;
#endif
constexpr bool kNumericStable =
#ifdef NUMERIC_STABLE
    true;
#else
    false;
#endif

// Exp stage is fused into the mask-add chain only when !NUMERIC_STABLE (otherwise exp happens in
// calc_numeric_stable). The empty struct carries no chain tag, so the chain skips it entirely.
struct NoOpChainElement {};
using OptionalExp = std::conditional_t<
    kNumericStable,
    NoOpChainElement,
    compute_kernel_lib::Exp<
        static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
        compute_kernel_lib::Approx::Exact,
        compute_kernel_lib::Dst::D0>>;

// Templated on the CBs so the eltwise_chain below can take them as compile-time NTTPs.
template <uint32_t block_w, uint32_t cb_in, uint32_t cb_max_scaler, uint32_t cb_max, uint32_t cb_out>
ALWI void calc_numeric_stable() {
    auto cb_max_obj = CircularBuffer(cb_max);
    auto cb_out_obj = CircularBuffer(cb_out);

    // Use reduce_helpers for MAX reduce (REDUCE_ROW, PRELOADED mode)
    // Note: The library handles waiting for scaler tile internally
    compute_kernel_lib::
        reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
            cb_in, cb_max_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::row(block_w));

    // x - max(x) then exp, fused. cb_in resident (popped at chain end) -> DeferredPop + Block;
    // cb_max (chain manages wait/pop) -> Bulk + Scalar; cb_out reserve+push block_w -> Bulk.
    // sub_bcast_cols_init_short -> BinaryDataFormatReconfig::Input; plain pack_tile
    // (format already cb_out) -> PackTileReconfig::None.
    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(block_w),
        compute_kernel_lib::BinaryFpu<
            cb_in,
            cb_max,
            compute_kernel_lib::BinaryFpuOp::Sub,
            compute_kernel_lib::BroadcastDim::Col,
            compute_kernel_lib::InputLifecycle::DeferredPop,
            compute_kernel_lib::InputLifecycle::Bulk,
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
    cb_out_obj.wait_front(block_w);
}

void kernel_main() {
    constexpr uint32_t block_h = get_compile_time_arg_val(0);
    constexpr uint32_t block_w = get_compile_time_arg_val(1);
    [[maybe_unused]] constexpr uint32_t subblock_w = get_compile_time_arg_val(2);
    [[maybe_unused]] constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(3);

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

    for (uint32_t i = 0; i < block_h; i++) {
#if FUSED_SCALE_MASK
        // fused scale: cb_in0 (resident sharded, popped at chain end) * cb_fused_scale (held
        // scalar) -> cb_scale_mask. cb_in0 DeferredPop + Block; cb_fused_scale CallerManaged +
        // Scalar (wait kept below); cb_scale_mask reserve+push block_w -> Bulk.
        // reconfig + mul_tiles_bcast_scalar_init_short -> Input; pack_reconfig -> Output.
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
            compute_kernel_lib::OperandKind::Scalar>(block_w);

        // fused mask add (+exp): cb_scale_mask (Bulk: wait+pop block_w) + cb_fused_attn
        // (lifecycle per the kCausalMask / kShardedCausalMask matrix — chain owns wait+pop,
        // exactly reproducing the original's #ifndef SHARDED_CAUSAL_MASK wait + #ifdef CAUSAL_MASK
        // pop) -> cb_x (reserve+push block_w -> Bulk). add init -> Input; plain pack_tile -> None.
        // Exp dropped when NUMERIC_STABLE (done in calc_numeric_stable below).
        constexpr auto fused_broadcast =
            kCausalMask ? compute_kernel_lib::BroadcastDim::None : compute_kernel_lib::BroadcastDim::Row;
        constexpr auto fused_attn_lifecycle =
            (kCausalMask && kShardedCausalMask) ? compute_kernel_lib::InputLifecycle::DeferredPop  // no wait, pop
            : kCausalMask                       ? compute_kernel_lib::InputLifecycle::Bulk         // wait + pop
            : !kShardedCausalMask               ? compute_kernel_lib::InputLifecycle::HeldBulk  // wait, no pop (held)
                                                : compute_kernel_lib::InputLifecycle::CallerManaged;          // no wait, no pop
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::EltwiseShape::tiles(block_w),
            compute_kernel_lib::BinaryFpu<
                cb_scale_mask,
                cb_fused_attn,
                compute_kernel_lib::BinaryFpuOp::Add,
                fused_broadcast,
                compute_kernel_lib::InputLifecycle::Bulk,
                fused_attn_lifecycle,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::OperandKind::Block>{},
            OptionalExp{},
            compute_kernel_lib::PackTile<
                cb_x,
                compute_kernel_lib::OutputLifecycle::Bulk,
                compute_kernel_lib::PackTileReconfig::None>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        cb_x_obj.wait_front(block_w);
        calc_numeric_stable<block_w, cb_x, cb_max_scaler, cb_max, cb_exps>();
#endif

        // cb_fused_attn pop is now owned by the fused-attn chain above (DeferredPop/Bulk under
        // CAUSAL_MASK; held otherwise) — no external pop needed.
        reconfig_data_format(cb_exps, cb_sum_scaler);

#else

#ifdef NUMERIC_STABLE
        calc_numeric_stable<block_w, cb_in0, cb_max_scaler, cb_max, cb_exps>();
#else
        // exp(x): CopyTile + Exp + PackTile chain over all block_w tiles (eliminates subblock loop).
        // cb_in0 wait+pop all block_w via Bulk lifecycle; cb_exps reserve+push -> Bulk.
        // Reconfig: reconfig_data_format(cb_in0, cb_in0) + pack_reconfig_data_format(cb_exps)
        // ONCE outside the outer loop — chain emits per-call (fold-elided after first).
        // -> CopyTileReconfig::Input + PackTileReconfig::Output.
        compute_kernel_lib::eltwise_chain(
            block_w,
            compute_kernel_lib::CopyTile<
                cb_in0,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::Bulk,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::OperandKind::Block>{},
            compute_kernel_lib::Exp<
                static_cast<compute_kernel_lib::Approx>(EXP_APPROX),
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<cb_exps, compute_kernel_lib::OutputLifecycle::Bulk>{});
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

        // exp(x) / (sum(exp(x))) — bcast COL on cb_recipsumexps (1 tile held, chain manages wait/pop).
        // Original per-subblock DEST batching with sequential index becomes
        // per-tile streaming via chain InputLifecycle::Streaming + Scalar on cb_exps.
        //
        // Reconfig: reconfig_data_format + mul_bcast_cols_init_short -> Input;
        // pack_reconfig_data_format(cb_out0) -> PackTileReconfig::Output.
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::EltwiseShape::tiles(block_w),
            compute_kernel_lib::BinaryFpu<
                cb_exps,
                cb_recipsumexps,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::Col,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::Bulk,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::PackTile<
                cb_out0,
                compute_kernel_lib::OutputLifecycle::Bulk,
                compute_kernel_lib::PackTileReconfig::Output>{});
    }
}
