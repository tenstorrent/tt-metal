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

// for scale+mask+softmax:
// bcast HW (mul by 1 tile)  example: (  [2,1,1024,64] * [1,1,32,32]  )
// bcast add H               example: ( [2,1,1024,64] + [2,1,32,64] ) (bcast W -> H)
// Note that the attention mask will not fit in L1 for the entire tensor
// The buffer for the att mask is currently sized as (1t,Wt) so we only reuse it for one HtWt-sized batch of x
// then read another Wt tiles of mask for the next batch

template <std::uint32_t dfb_in, std::uint32_t dfb_max_scaler, std::uint32_t dfb_max, std::uint32_t dfb_out>
void calc_numeric_stable(std::uint32_t Wt, std::uint32_t ndst) {
    DataflowBuffer dfb_out_obj(dfb_out);

    // calculate max val per row
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        dfb_in,
        dfb_max_scaler,
        dfb_max,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));

    // calculate x-max(x)
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(Wt).block_size(ndst),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Sub,
            ckl::input(
                dfb_in,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::input(dfb_max, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{},
        ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>{},
        // reuse the exps buffer again, this time in a circular manner
        ckl::PackTile<ckl::output(
            dfb_out,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
    dfb_out_obj.wait_front(static_cast<uint16_t>(Wt));
}

// CB consumers cannot wrap mid-fifo: pops in one cycle must land exactly on fifo_limit.
// After a partial row (Wt not a multiple of the CB capacity), rd/wr sit at that offset.
// Push/pop `pad` tiles to complete the cycle and return pointers to the CB base.
// Kept identical to the copy in softmax_large_tensor.cpp.
ALWI void cycle_dfb_pad(std::uint32_t dfb_id, std::uint32_t pad) {
    if (pad == 0) {
        return;
    }
    DataflowBuffer dfb(static_cast<uint16_t>(dfb_id));
    dfb.reserve_back(static_cast<uint16_t>(pad));
    dfb.push_back(static_cast<uint16_t>(pad));
    dfb.wait_front(static_cast<uint16_t>(pad));
    dfb.pop_front(static_cast<uint16_t>(pad));
}

// Same, for CBs whose padding tiles the reader already pushed: only consume them.
ALWI void drain_dfb_pad(std::uint32_t dfb_id, std::uint32_t pad) {
    if (pad == 0) {
        return;
    }
    DataflowBuffer dfb(static_cast<uint16_t>(dfb_id));
    dfb.wait_front(static_cast<uint16_t>(pad));
    dfb.pop_front(static_cast<uint16_t>(pad));
}

void kernel_main() {
    const std::uint32_t NCHt = get_arg(args::num_rows);
    const std::uint32_t Ht = get_arg(args::Ht);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t ndst = get_arg(args::blk);
    const std::uint32_t start_ht = get_arg(args::start_ht);
    // The pad-mask data (W > W_unpadded) is host-known; it is carried as the MASK_PADDED_DATA compile-time
    // define (not a runtime arg), which lets the c_5 pad-mask DFB and the c_10 intermediate be bound only
    // on the paths that use them.
    constexpr std::uint32_t in0_t = get_arg(args::in0_t);          // in0 DFB tile capacity
    const std::uint32_t out0_t = ndst * 2;                         // matches factory out0_t = block_size * 2
    const std::uint32_t exps_t = ((Wt + ndst - 1) / ndst) * ndst;  // dfb_exps/dfb_x capacity, rounded to ndst
    // Every CB capacity is a multiple of ndst, so when ndst divides Wt the blocks tile each fifo
    // exactly: a row already ends on the base and no pad is needed. Zero guards keep the modulo safe.
    const bool pad_to_fifo_base = Wt > 0 && ndst > 0 && (Wt % ndst) != 0;
    // Tiles needed after Wt to finish each CB's cycle, named for the capacity they align.
    const std::uint32_t in0_pad = pad_to_fifo_base ? ((in0_t - (Wt % in0_t)) % in0_t) : 0;
    const std::uint32_t out0_pad = pad_to_fifo_base ? ((out0_t - (Wt % out0_t)) % out0_t) : 0;
    const std::uint32_t exps_pad = pad_to_fifo_base ? (exps_t - Wt) : 0;
    const std::uint32_t attn_pad = exps_pad;  // in4_t is also round_up(Wt, ndst); reader pushes the pad
    const std::uint32_t scale_mask_pad = pad_to_fifo_base ? (exps_pad + ndst) : 0;  // im3_t = exps_t + ndst

    constexpr std::uint32_t onetile = 1;
    // reserve one tile for zeros on dfb_in2
    // We only do the reserve for the intermediates once and use pack_tile
    // So effectively these are used as pre-allocated arrays
    // Note that the entire W dimension must fit in the intermed0 CB for this kernel to be correct
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;
    constexpr auto dfb_exps = dfb::exps;
    constexpr auto dfb_recipsumexps = dfb::recip_sum_exps;
    constexpr auto dfb_in0 = dfb::in0;
    constexpr auto dfb_out0 = dfb::out0;
#ifdef FUSED_SCALE_MASK
    constexpr auto dfb_fused_scale = dfb::fused_scale;
#endif
#ifdef FUSED_SCALE_MASK
    constexpr auto dfb_fused_attn = dfb::fused_attn;
#endif
#ifdef FUSED_SCALE_MASK
    constexpr auto dfb_scale_mask = dfb::scale_mask;
#endif
#ifdef MASK_PADDED_DATA
    constexpr auto dfb_mask_padded = dfb::mask_padded;
#endif
#ifdef NUMERIC_STABLE
    constexpr auto dfb_max = dfb::max;
#endif
    DataflowBuffer dfb_max_scaler_obj(dfb_max_scaler);
    DataflowBuffer dfb_sum_scaler_obj(dfb_sum_scaler);
    DataflowBuffer dfb_out0_obj(dfb_out0);
#ifdef FUSED_SCALE_MASK
    // fused_scale/fused_attn/scale_mask are bound only on the fused scale-mask path.
    DataflowBuffer dfb_fused_scale_obj(dfb_fused_scale);
    DataflowBuffer dfb_fused_attn_obj(dfb_fused_attn);
#endif
    compute_kernel_hw_startup(dfb_in0, dfb_max_scaler, dfb_exps);
#ifdef NUMERIC_STABLE
#if defined(FUSED_SCALE_MASK) || defined(MASK_PADDED_DATA)
    // dfb_x is a distinct intermediate (c_10) only on the numeric-stable paths that post-process a masked
    // buffer; otherwise the reads go straight from dfb_in0 (see the calc_numeric_stable<dfb_in0,...> call).
    constexpr auto dfb_x = dfb::x;
#endif
#else
    // Without numeric_stable, dfb_x aliases dfb_exps (Same-FIFO reuse) so exp results circulate in one buffer.
    constexpr auto dfb_x = dfb_exps;
#endif

    dfb_max_scaler_obj.wait_front(1);  // comes from the reader
    dfb_sum_scaler_obj.wait_front(1);  // comes from the reader

#ifdef FUSED_SCALE_MASK
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
    for (std::uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef FUSED_SCALE_MASK
        // apply fused scale [*= 1/sqrt(...)]
        ckl::mul<
            ckl::input(
                dfb_in0, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::input(dfb_fused_scale, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            // reuse exps buffer
            ckl::output(dfb_scale_mask, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::tiles(Wt).block_size(ndst));
#ifndef CAUSAL_MASK
        if (wait_mask) {
            dfb_fused_attn_obj.wait_front(Wt);
        }
#endif
        constexpr auto mask_bcast = causal_mask ? ckl::BroadcastDim::None : ckl::BroadcastDim::Row;
        constexpr auto attn_wait = causal_mask ? ckl::WaitPolicy::Cumulative : ckl::WaitPolicy::None;
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(Wt).block_size(ndst),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(
                    dfb_scale_mask,
                    ckl::WaitPolicy::PerBlockSize,
                    ckl::PopPolicy::PerBlockSize,
                    ckl::InputTileMapping::Block),
                ckl::input(
                    dfb_fused_attn, mask_bcast, attn_wait, ckl::PopPolicy::None, ckl::InputTileMapping::Block)>{},
            ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
            // reuse the exps buffer again, this time in a circular manner
            ckl::PackTile<ckl::output(
                dfb_x,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
        calc_numeric_stable<dfb_x, dfb_max_scaler, dfb_max, dfb_exps>(Wt, ndst);
#endif

#ifdef CAUSAL_MASK
        dfb_fused_attn_obj.pop_front(Wt);
        drain_dfb_pad(dfb_fused_attn, attn_pad);
#else
        if (wait_mask) {
            wait_mask = false;
        }
        ht++;
        if (ht == Ht) {
            dfb_fused_attn_obj.pop_front(Wt);
            drain_dfb_pad(dfb_fused_attn, attn_pad);
            ht = 0;
            wait_mask = true;
        }
#endif  // CAUSAL_MASK

        reconfig_data_format(dfb_exps, dfb_sum_scaler);
#else
        reconfig_data_format(dfb_in0, dfb_in0);
        pack_reconfig_data_format(dfb_exps);
        copy_init(dfb_in0);  // need to copy from CB to DST to be able to run sfpu math
#ifndef NUMERIC_STABLE
        exp_tile_init<EXP_APPROX>();
#endif
#ifdef MASK_PADDED_DATA
        {
            if (Wt > 1) {
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(Wt - 1).block_size(ndst),
                    ckl::CopyTile<ckl::input(
                        dfb_in0,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block)>{},
                    ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
                    ckl::PackTile<ckl::output(
                        dfb_x,
                        ckl::ReservePolicy::PerBlockSize,
                        ckl::PushPolicy::PerBlockSize,
                        ckl::DataFormatReconfig::Disabled)>{});
            }

            // last tile of the row gets the -inf padding mask
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Add,
                    ckl::input(dfb_in0),
                    ckl::input(
                        dfb_mask_padded,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None)>{},  // dfb_mask_padded: held scalar, chain waits(1), no
                                                   // pop
                ckl::Optional<!numeric_stable, ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>>{},
                ckl::PackTile<ckl::output(
                    dfb_x,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>{});

// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<dfb_x, dfb_max_scaler, dfb_max, dfb_exps>(Wt, ndst);
#endif
        }
#else
        {
// add numeric_stable
// fuse exp with sub tiles
#ifdef NUMERIC_STABLE
            calc_numeric_stable<dfb_in0, dfb_max_scaler, dfb_max, dfb_exps>(Wt, ndst);
#else
            ckl::unary<
                ckl::Exp<static_cast<ckl::Approx>(EXP_APPROX), ckl::Dst::D0>,
                ckl::input(
                    dfb_in0, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
                ckl::output(
                    dfb_exps,
                    ckl::ReservePolicy::PerBlockSize,
                    ckl::PushPolicy::PerBlockSize,
                    ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::tiles(Wt).block_size(ndst));
#endif
        }
#endif  // MASK_PADDED_DATA
#endif  // FUSED_SCALE_MASK

        // SUM reduce with reciprocal post-processing (1/sum)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb_exps,
            dfb_sum_scaler,
            dfb_recipsumexps,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
            compute_kernel_lib::ReduceInputBlockShape::row(Wt),
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

        // tile *= 1/(sum(exp(x)))
        ckl::mul<
            ckl::input(dfb_exps, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
            ckl::input(dfb_recipsumexps, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::output(dfb_out0, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::tiles(Wt).block_size(ndst));

        // Realign CBs before the next row when Wt does not fill them exactly.
        drain_dfb_pad(dfb_in0, in0_pad);
        cycle_dfb_pad(dfb_exps, exps_pad);
#ifdef FUSED_SCALE_MASK
        cycle_dfb_pad(dfb_scale_mask, scale_mask_pad);
#ifdef NUMERIC_STABLE
        // Without NUMERIC_STABLE, dfb_x aliases dfb_exps; cycling it again would drift it per row.
        cycle_dfb_pad(dfb_x, exps_pad);
#endif
#elif defined(NUMERIC_STABLE) && defined(MASK_PADDED_DATA)
        // dfb_x is a distinct buffer only here; without NUMERIC_STABLE it aliases dfb_exps (already cycled).
        cycle_dfb_pad(dfb_x, exps_pad);
#endif
        if (out0_pad > 0) {
            dfb_out0_obj.reserve_back(static_cast<uint16_t>(out0_pad));
            dfb_out0_obj.push_back(static_cast<uint16_t>(out0_pad));  // writer drains, does not write to DRAM
        }
    }  // NCHt loop
    // The scaler tiles are each waited once and reused across the whole NCHt loop; pop them at
    // the end so the CBs are left balanced.
    dfb_max_scaler_obj.pop_front(1);
    dfb_sum_scaler_obj.pop_front(1);
#ifdef FUSED_SCALE_MASK
    dfb_fused_scale_obj.pop_front(1);
#endif
}
