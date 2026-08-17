// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
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
    constexpr auto dfb_in0_id = dfb::in0;
    constexpr auto dfb_mask_id = dfb::mask;
    DataflowBuffer dfb_mask_obj(dfb_mask_id);
    constexpr auto dfb_max_scaler_id = dfb::max_scaler;
    DataflowBuffer dfb_max_scaler_obj(dfb_max_scaler_id);
    constexpr auto dfb_sum_scaler_id = dfb::sum_scaler;
    DataflowBuffer dfb_sum_scaler_obj(dfb_sum_scaler_id);
    constexpr auto dfb_out0_id = dfb::out0;
    constexpr auto dfb_exps_id = dfb::exps;
    constexpr auto dfb_recipsumexps_id = dfb::recip_sum_exps;
    constexpr auto dfb_max_id = dfb::max;
    constexpr auto dfb_x_m_max_id = dfb::x_minus_max;
    DataflowBuffer dfb_x_m_max_obj(dfb_x_m_max_id);
    constexpr auto dfb_tmp_id = dfb::tmp;

    compute_kernel_hw_startup(dfb_in0_id, dfb_max_scaler_id, dfb_out0_id);

    constexpr uint32_t onetile = 1;

    uint32_t N = get_arg(args::N);
    uint32_t Wt = get_arg(args::Wt);

    dfb_mask_obj.wait_front(onetile);
    dfb_max_scaler_obj.wait_front(onetile);
    dfb_sum_scaler_obj.wait_front(onetile);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Wt == 1) {
            mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_tmp_id>(0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb_tmp_id, dfb_max_scaler_id, dfb_max_id>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into dfb_max_id via the helper.
            // dfb_in0_id holds all Wt tiles persistently for later steps, so use
            // WaitUpfrontNoPop — the helper waits for the slice it needs and never pops.
            ckl::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                dfb_in0_id,
                dfb_max_scaler_id,
                dfb_max_id,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile (index Wt-1, no pop) and continue reducing
            // into dfb_max_id via Accumulate. The accumulator and output are both dfb_max_id:
            // the helper waits+pops the previous tile, then packs+pushes the new one.
            mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_tmp_id>(Wt - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb_tmp_id, dfb_max_scaler_id, dfb_max_id>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(dfb_max_id, /*iter=*/1));
        }

        ckl::sub<
            ckl::input(
                dfb_in0_id,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::OperandKind::Block,
                kDataFormatReconfig),
            ckl::input(
                dfb_max_id,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                kDataFormatReconfig),
            ckl::output(dfb_x_m_max_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Wt));

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
                    dfb_x_m_max_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Block,
                    kDataFormatReconfig),
                ckl::Dst::D0>{},
            ckl::Optional<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_exps_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    dfb_x_m_max_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Block,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::Dst::D0>{Wt - 1},
            ckl::Optional<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::CopyTile<
                ckl::input(dfb_mask_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::Dst::D1>{},
            ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_exps_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

#ifdef LOG
        // log(sum) - pop tiles after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            dfb_exps_id,
            dfb_sum_scaler_id,
            dfb_recipsumexps_id,
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
            dfb_exps_id,
            dfb_sum_scaler_id,
            dfb_recipsumexps_id,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::row(Wt),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        dfb_x_m_max_obj.wait_front(Wt);
#ifdef LOG
        ckl::sub<
            ckl::input(
                dfb_x_m_max_id,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::OperandKind::Block,
                kDataFormatReconfig),
            ckl::input(
                dfb_recipsumexps_id,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                kDataFormatReconfig),
            ckl::output(dfb_out0_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Wt));
#else
        ckl::mul<
            ckl::input(
                dfb_exps_id,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::OperandKind::Block,
                kDataFormatReconfig),
            ckl::input(
                dfb_recipsumexps_id,
                ckl::BroadcastDim::Col,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                kDataFormatReconfig),
            ckl::output(dfb_out0_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Wt));
#endif
        dfb_x_m_max_obj.pop_front(Wt);
    }
}
