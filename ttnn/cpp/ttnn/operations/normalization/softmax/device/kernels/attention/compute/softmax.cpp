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

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

template <uint32_t cb_in, uint32_t cb_max_scaler, uint32_t cb_max, uint32_t cb_out>
void calc_numeric_stable(uint32_t Wt, uint32_t ndst) {
    auto cb_out_obj = CircularBuffer(cb_out);

    // calculate max val per row
    ckl::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        cb_in,
        cb_max_scaler,
        cb_max,
        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
        ckl::ReduceDataFormatReconfigMode::INPUT>(ckl::ReduceInputBlockShape::row(Wt));

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(Wt, ndst),
        ckl::BinaryFpu<
            ckl::input(cb_in, ckl::InputLifecycle::DeferredPop, ckl::OperandKind::Block),
            ckl::input(cb_max, ckl::InputLifecycle::Bulk),
            ckl::BinaryFpuOp::Sub,
            ckl::BroadcastDim::Col>{},
        ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Approx::Exact, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(cb_out, ckl::OutputLifecycle::Chunked, ckl::DataFormatReconfig::Disabled)>{});
    cb_out_obj.wait_front(Wt);
}

void kernel_main() {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    const uint32_t Ht = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t ndst = get_arg_val<uint32_t>(3);
    const uint32_t start_ht = get_arg_val<uint32_t>(4);
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(5);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_6);

    constexpr uint32_t onetile = 1;
    // reserve one tile for zeros on cb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_13;
    constexpr auto cb_fused_scale = tt::CBIndex::c_3;
    constexpr auto cb_fused_attn = tt::CBIndex::c_4;
    constexpr auto cb_mask_padded = tt::CBIndex::c_5;
    constexpr auto cb_exps = tt::CBIndex::c_6;
    constexpr auto cb_scale_mask = tt::CBIndex::c_9;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_7;
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_11;
    CircularBuffer cb_max_scaler_obj(cb_max_scaler);
    CircularBuffer cb_sum_scaler_obj(cb_sum_scaler);
    CircularBuffer cb_fused_scale_obj(cb_fused_scale);
    CircularBuffer cb_fused_attn_obj(cb_fused_attn);
    CircularBuffer cb_scale_mask_obj(cb_scale_mask);
    CircularBuffer cb_in0_obj(cb_in0);
    CircularBuffer cb_out0_obj(cb_out0);
#ifdef NUMERIC_STABLE
    constexpr auto cb_max = tt::CBIndex::c_8;
    constexpr auto cb_x = tt::CBIndex::c_10;
    CircularBuffer cb_max_obj(cb_max);
#else
    constexpr auto cb_x = cb_exps;
#endif
    CircularBuffer cb_x_obj(cb_x);

    cb_max_scaler_obj.wait_front(1);  // comes from the reader
    cb_sum_scaler_obj.wait_front(1);  // comes from the reader

#if FUSED_SCALE_MASK
    cb_fused_scale_obj.wait_front(1);
#endif

    constexpr int dst0 = 0;
    uint32_t ht = start_ht;
    bool wait_mask = true;
#ifdef CAUSAL_MASK
    [[maybe_unused]] constexpr bool causal_mask = true;
#else
    [[maybe_unused]] constexpr bool causal_mask = false;
#endif
#ifdef NUMERIC_STABLE
    [[maybe_unused]] constexpr bool numeric_stable = true;
#else
    [[maybe_unused]] constexpr bool numeric_stable = false;
#endif
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#if FUSED_SCALE_MASK
        ckl::mul<
            ckl::input(cb_in0),
            ckl::input(cb_fused_scale, ckl::InputLifecycle::CallerManaged),
            ckl::output(cb_scale_mask),
            ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::tiles(Wt));
#ifdef CAUSAL_MASK
        cb_fused_attn_obj.wait_front(Wt);
#else
        if (wait_mask) {
            cb_fused_attn_obj.wait_front(Wt);
        }
#endif
        constexpr auto mask_bcast = causal_mask ? ckl::BroadcastDim::None : ckl::BroadcastDim::Row;
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(Wt, ndst),
            ckl::BinaryFpu<
                ckl::input(cb_scale_mask, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
                ckl::input(cb_fused_attn, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block),
                ckl::BinaryFpuOp::Add,
                mask_bcast>{},
            ckl::OptionalChainElement<
                !numeric_stable,
                ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Approx::Exact, ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(cb_x, ckl::OutputLifecycle::Chunked, ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        calc_numeric_stable<cb_x, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
#endif

#ifdef CAUSAL_MASK
        cb_fused_attn_obj.pop_front(Wt);
#else
        if (wait_mask) {
            wait_mask = false;
        }
        ht++;
        if (ht == Ht) {
            cb_fused_attn_obj.pop_front(Wt);
            ht = 0;
            wait_mask = true;
        }
#endif  // CAUSAL_MASK

        reconfig_data_format(cb_exps, cb_sum_scaler);
#else
        reconfig_data_format(cb_in0, cb_in0);
        pack_reconfig_data_format(cb_exps);
        copy_tile_to_dst_init_short(cb_in0);  // need to copy from CB to DST to be able to run sfpu math
#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
        if (mask_padded_data) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(Wt - 1),
                ckl::CopyTile<ckl::input(cb_in0)>{},
                ckl::OptionalChainElement<
                    !numeric_stable,
                    ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Approx::Exact, ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(cb_x, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});

            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::BinaryFpu<
                    ckl::input(cb_in0),
                    ckl::input(cb_mask_padded, ckl::InputLifecycle::HeldBulk),
                    ckl::BinaryFpuOp::Add,
                    ckl::BroadcastDim::Row>{},  // cb_mask_padded: held scalar, chain waits(1), no
                                                // pop
                ckl::OptionalChainElement<
                    !numeric_stable,
                    ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Approx::Exact, ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(cb_x, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<cb_x, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
#endif

        } else {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<cb_in0, cb_max_scaler, cb_max, cb_exps>(Wt, ndst);
#else
            ckl::unary<
                ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Approx::Exact, ckl::Dst::D0>,
                ckl::input(cb_in0),
                ckl::output(cb_exps, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(Wt));
#endif
        }
#endif

        // SUM reduce with reciprocal post-processing (1/sum)
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::row(Wt),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t) {
                recip_tile_init();
                recip_tile(0);
            });

        ckl::mul<
            ckl::input(cb_exps, ckl::InputLifecycle::DeferredPop, ckl::OperandKind::Block),
            ckl::input(cb_recipsumexps, ckl::InputLifecycle::Bulk),
            ckl::output(cb_out0, ckl::OutputLifecycle::Chunked),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(Wt, ndst));
    }  // NCHt loop
    // The scaler tiles are each waited once and reused across the whole NCHt loop; pop them at
    // the end so the CBs are left balanced.
    cb_max_scaler_obj.pop_front(1);
    cb_sum_scaler_obj.pop_front(1);
#if FUSED_SCALE_MASK
    cb_fused_scale_obj.pop_front(1);
#endif
}
