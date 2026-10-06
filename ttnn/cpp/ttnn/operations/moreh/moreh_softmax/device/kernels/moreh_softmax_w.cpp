// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // sub
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"       // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"       // Mask, Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    DataflowBuffer dfb_mask_obj(dfb::mask);
    DataflowBuffer dfb_max_scaler_obj(dfb::max_scaler);
    DataflowBuffer dfb_sum_scaler_obj(dfb::sum_scaler);
    DataflowBuffer dfb_x_m_max_obj(dfb::x_minus_max);

    compute_kernel_hw_startup(dfb::in0, dfb::max_scaler, dfb::out0);

    constexpr uint32_t onetile = 1;

    // Plain uint32_t (not constexpr) to match legacy get_compile_time_arg_val typing and avoid
    // force-unrolling the per-Wt loops (see moreh_softmax_w_large.cpp for the LTO/addrmod rationale).
    const std::uint32_t N = get_arg(args::N);
    const std::uint32_t Wt = get_arg(args::Wt);

    dfb_mask_obj.wait_front(onetile);
    dfb_max_scaler_obj.wait_front(onetile);
    dfb_sum_scaler_obj.wait_front(onetile);

    for (std::uint32_t n = 0; n < N; ++n) {
        // SOFTMIN shift: use min(x) instead of max(x). A max-based shift is +inf for a row
        // holding +inf, and exp(max(x) - x) then saturates the whole row (inf / inf = NaN, and the
        // +inf lane itself is inf - inf = NaN) -- see #56371. Shifted by min(x) instead, a row that
        // holds +inf still has a finite minimum, so the +inf lanes exponentiate to 0 and the finite
        // lanes keep their relative weights -- that is torch's distribution. (A row whose minimum is
        // itself -inf is the documented divergent case: the whole mass lands on that lane.)
        // The padding lanes of the last tile are masked to +inf: a 0-masked pad (what the MAX path
        // wants) would win the MIN on an all-positive row.
#ifdef SOFTMAX
        // find max value
        if (Wt == 1) {
            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb::tmp, dfb::max_scaler, dfb::max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into dfb::max via the helper.
            // dfb::in0 holds all Wt tiles persistently for later steps, so use
            // WaitUpfrontNoPop — the helper waits for the slice it needs and never pops.
            ckl::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                dfb::in0,
                dfb::max_scaler,
                dfb::max,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile (index Wt-1, no pop) and continue reducing
            // into dfb::max via Accumulate. The accumulator and output are both dfb::max:
            // the helper waits+pops the previous tile, then packs+pushes the new one.
            mask_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(Wt - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb::tmp, dfb::max_scaler, dfb::max>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(dfb::max, /*iter=*/1));
        }
#else
        // MIN is SFPU-only: a Fast fp32 MIN does not exist, so an fp32 reduce input needs
        // ReduceFp32Mode::Accurate; bf16 ignores fp32_mode entirely. Deduce it from the compile-time
        // formats of the buffers THIS reduce can read (unpack_src_format[] is a constexpr descriptor
        // array, so it is usable as a template argument):
        //   * dfb::in0 -- the leading tiles (the large kernels' phase 1 reduces them straight out of it);
        //   * dfb::tmp -- the masked last tile; the factories declare TMP with intermed_data_format,
        //                 which is Float32 whenever fp32_dest_acc_en is enabled, even for a bf16 tensor.
        // Deriving the mode from dfb::in0 alone picks Fast for bf16 + fp32 accumulation and trips the
        // MIN static_assert in reduce_helpers_compute.inl.
        //   * dfb::x_minus_max -- the staged copy the reduce below actually reads when Wt/Ht > 1.
        constexpr DataFormat kSoftminStagedFormat = static_cast<DataFormat>(unpack_src_format[dfb::x_minus_max]);
        constexpr DataFormat kSoftminInFormat = static_cast<DataFormat>(unpack_src_format[dfb::in0]);
        constexpr DataFormat kSoftminScratchFormat = static_cast<DataFormat>(unpack_src_format[dfb::tmp]);
        constexpr ReduceFp32Mode kFp32Mode =
            (kSoftminInFormat == DataFormat::Float32 || kSoftminScratchFormat == DataFormat::Float32 || kSoftminStagedFormat == DataFormat::Float32)
                ? ReduceFp32Mode::Accurate
                : ReduceFp32Mode::Fast;
        // find min value
        if (Wt == 1) {
            mask_posinf_tile_to_dfb<dfb::in0, dfb::mask, dfb::tmp>(0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<
                PoolType::MIN,
                ReduceDim::REDUCE_ROW,
                dfb::tmp,
                dfb::max_scaler,
                dfb::max,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kFp32Mode>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Stage a masked copy of the whole reduce axis in dfb::x_minus_max (Wt entries: Wt - 1 plain copies
            // plus the masked last tile, so the staging buffer has to be Wt deep; still free here)
            // and reduce it in ONE call, instead of running a second reduce() over the masked tile and
            // reloading the first one as an accumulator. The reload copies back a tile that the packer
            // wrote with the reduce mask applied, so only the reduce lane is valid there and every
            // other lane comes back as 0; an element-wise MIN fold would take those 0s as real input
            // and clamp every row minimum to 0 whenever the row is all-positive. (The same reload is
            // only accidentally right for MAX on non-negative data and for SUM, whose identity is 0.)
            for (uint32_t i = 0; i < Wt - 1; ++i) {
                copy_tile_to_dfb<dfb::in0, dfb::x_minus_max>(i, /*pop=*/0);
            }
            mask_posinf_tile_to_dfb<dfb::in0, dfb::mask, dfb::x_minus_max>(Wt - 1, 0, /*pop0=*/0, /*popm=*/0);
            ckl::reduce<
                PoolType::MIN,
                ReduceDim::REDUCE_ROW,
                dfb::x_minus_max,
                dfb::max_scaler,
                dfb::max,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                kFp32Mode>(
                ckl::ReduceInputBlockShape::row(Wt));
        }
#endif

        // compute x - max(x)
        ckl::sub<
            ckl::input(
                dfb::in0,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::InputTileMapping::Block,
                kDataFormatReconfig),
            ckl::input(
                dfb::max, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, kDataFormatReconfig),
            ckl::output(dfb::x_minus_max, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Wt));

        // compute exp(x - max(x))
        dfb_x_m_max_obj.wait_front(Wt);
#ifdef SOFTMAX
        constexpr bool is_softmax = true;
#else
        constexpr bool is_softmax = false;
#endif
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(Wt - 1),
            ckl::CopyTile<
                ckl::input(
                    dfb::x_minus_max,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Block,
                    kDataFormatReconfig),
                ckl::Dst::D0>{},
            ckl::Optional<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb::exps, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    dfb::x_minus_max,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Block,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset),
                ckl::Dst::D0>{Wt - 1},
            ckl::Optional<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::CopyTile<
                ckl::input(dfb::mask, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::Dst::D1>{},
            ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb::exps, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

#ifdef LOG
        // log(sum) - pop tiles after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb::exps,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::row(Wt),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        // 1/sum - keep tiles for subsequent multiplication
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb::exps,
            dfb::sum_scaler,
            dfb::recip_sum_exps,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::row(Wt),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // compute final result
        dfb_x_m_max_obj.wait_front(Wt);
#ifdef LOG
        ckl::sub<
            ckl::input(
                dfb::x_minus_max,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block,
                kDataFormatReconfig),
            ckl::input(
                dfb::recip_sum_exps,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                kDataFormatReconfig),
            ckl::output(dfb::out0, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Wt));
#else
        ckl::mul<
            ckl::input(
                dfb::exps,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::InputTileMapping::Block,
                kDataFormatReconfig),
            ckl::input(
                dfb::recip_sum_exps,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                kDataFormatReconfig),
            ckl::output(dfb::out0, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Wt));
#endif
        dfb_x_m_max_obj.pop_front(Wt);
    }
}
