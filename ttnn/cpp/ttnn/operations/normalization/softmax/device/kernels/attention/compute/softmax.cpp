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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

template <uint32_t dfb_in_id, uint32_t dfb_max_scaler_id, uint32_t dfb_max_id, uint32_t dfb_out_id>
void calc_numeric_stable(uint32_t Wt, uint32_t ndst) {
    auto dfb_out_obj_id = DataflowBuffer(dfb_out_id);

    // calculate max val per row
    ckl::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        dfb_in_id,
        dfb_max_scaler_id,
        dfb_max_id,
        ckl::ReduceInputPolicy::WaitUpfrontNoPop,
        ckl::ReduceDataFormatReconfigMode::INPUT>(ckl::ReduceInputBlockShape::row(Wt));

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(Wt, ndst),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Sub,
            ckl::input(dfb_in_id, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(dfb_max_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{},
        ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_out_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
    dfb_out_obj_id.wait_front(Wt);
}

void kernel_main() {
    const uint32_t NCHt = get_arg(args::num_rows);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t ndst = get_arg(args::blk);
    const uint32_t start_ht = get_arg(args::start_ht);

    constexpr std::uint32_t onetile = 1;
    // reserve one tile for zeros on dfb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 DFB for this kernel to be correct
    constexpr auto dfb_max_scaler_id = dfb::max_scaler;
    constexpr auto dfb_sum_scaler_id = dfb::sum_scaler;
    constexpr auto dfb_exps_id = dfb::exps;
    constexpr auto dfb_recipsumexps_id = dfb::recip_sum_exps;
    constexpr auto dfb_in0_id = dfb::in0;
    constexpr auto dfb_out0_id = dfb::out0;
    DataflowBuffer dfb_max_scaler_obj(dfb_max_scaler_id);
    DataflowBuffer dfb_sum_scaler_obj(dfb_sum_scaler_id);
    DataflowBuffer dfb_in0_obj(dfb_in0_id);
    DataflowBuffer dfb_out0_obj(dfb_out0_id);
#if FUSED_SCALE_MASK
    constexpr auto dfb_fused_scale_id = dfb::fused_scale;
    constexpr auto dfb_fused_attn_id = dfb::fused_attn;
    constexpr auto dfb_scale_mask_id = dfb::scale_mask;
    DataflowBuffer dfb_fused_scale_obj(dfb_fused_scale_id);
    DataflowBuffer dfb_fused_attn_obj(dfb_fused_attn_id);
    DataflowBuffer dfb_scale_mask_obj(dfb_scale_mask_id);
#endif
#ifdef MASK_PADDED_DATA
    constexpr auto dfb_mask_padded_id = dfb::mask_padded;
#endif

    compute_kernel_hw_startup(dfb_in0_id, dfb_max_scaler_id, dfb_exps_id);
#ifdef NUMERIC_STABLE
    constexpr auto dfb_max_id = dfb::max;
#if defined(FUSED_SCALE_MASK) || defined(MASK_PADDED_DATA)
    constexpr auto dfb_x_id = dfb::x;
    DataflowBuffer dfb_x_obj(dfb_x_id);
#endif
#else
    constexpr auto dfb_x_id = dfb_exps_id;
    DataflowBuffer dfb_x_obj(dfb_x_id);
#endif

    dfb_max_scaler_obj.wait_front(1);  // comes from the reader
    dfb_sum_scaler_obj.wait_front(1);  // comes from the reader

#if FUSED_SCALE_MASK
    dfb_fused_scale_obj.wait_front(1);
#endif

    constexpr int dst0 = 0;
    std::uint32_t ht = start_ht;
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
            ckl::input(dfb_in0_id),
            ckl::input(dfb_fused_scale_id, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::output(dfb_scale_mask_id)>(ckl::IterationShape::tiles(Wt));
#ifdef CAUSAL_MASK
        dfb_fused_attn_obj.wait_front(Wt);
#else
        if (wait_mask) {
            dfb_fused_attn_obj.wait_front(Wt);
        }
#endif
        constexpr auto mask_bcast = causal_mask ? ckl::BroadcastDim::None : ckl::BroadcastDim::Row;
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(Wt, ndst),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(
                    dfb_scale_mask_id,
                    ckl::WaitPolicy::PerBlockSize,
                    ckl::PopPolicy::PerBlockSize,
                    ckl::OperandKind::Block),
                ckl::input(
                    dfb_fused_attn_id,
                    mask_bcast,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Block)>{},
            ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(
                dfb_x_id,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        calc_numeric_stable<dfb_x_id, dfb_max_scaler_id, dfb_max_id, dfb_exps_id>(Wt, ndst);
#endif

#ifdef CAUSAL_MASK
        dfb_fused_attn_obj.pop_front(Wt);
#else
        if (wait_mask) {
            wait_mask = false;
        }
        ht++;
        if (ht == Ht) {
            dfb_fused_attn_obj.pop_front(Wt);
            ht = 0;
            wait_mask = true;
        }
#endif  // CAUSAL_MASK

        reconfig_data_format(dfb_exps_id, dfb_sum_scaler_id);
#else
        reconfig_data_format(dfb_in0_id, dfb_in0_id);
        pack_reconfig_data_format(dfb_exps_id);
        copy_tile_to_dst_init_short(dfb_in0_id);  // need to copy from DFB to DST to be able to run sfpu math
#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
#ifdef MASK_PADDED_DATA
        {
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(Wt - 1),
                ckl::CopyTile<ckl::input(dfb_in0_id)>{},
                ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(
                    dfb_x_id,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>{});

            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    ckl::input(dfb_in0_id),
                    ckl::input(
                        dfb_mask_padded_id,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None)>{},  // dfb_mask_padded_id: held scalar, chain waits(1), no
                                                   // pop
                ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(
                    dfb_x_id,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<dfb_x_id, dfb_max_scaler_id, dfb_max_id, dfb_exps_id>(Wt, ndst);
#endif
        }
#else
        {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<dfb_in0_id, dfb_max_scaler_id, dfb_max_id, dfb_exps_id>(Wt, ndst);
#else
            ckl::unary<
                ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>,
                ckl::input(dfb_in0_id),
                ckl::output(
                    dfb_exps_id,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::tiles(Wt));
#endif
        }
#endif  // MASK_PADDED_DATA
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal post-processing (1/sum)
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb_exps_id,
            dfb_sum_scaler_id,
            dfb_recipsumexps_id,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::row(Wt),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t) {
                recip_tile_init();
                recip_tile(0);
            });

        ckl::mul<
            ckl::input(dfb_exps_id, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(dfb_recipsumexps_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb_out0_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::tiles(Wt, ndst));
    }  // NCHt loop
    // The scaler tiles are each waited once and reused across the whole NCHt loop; pop them at
    // the end so the DFBs are left balanced.
    dfb_max_scaler_obj.pop_front(1);
    dfb_sum_scaler_obj.pop_front(1);
#if FUSED_SCALE_MASK
    dfb_fused_scale_obj.pop_front(1);
#endif
}
