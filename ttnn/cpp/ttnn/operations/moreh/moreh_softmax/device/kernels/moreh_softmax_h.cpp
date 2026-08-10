// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Mask, Negative
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

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb_in0_id, dfb_max_scaler_id, dfb_out0_id);

    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);

    dfb_mask_obj.wait_front(onetile);
    dfb_max_scaler_obj.wait_front(onetile);
    dfb_sum_scaler_obj.wait_front(onetile);

    for (std::uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Ht == 1) {
            mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_tmp_id>(0, 0, /*pop0=*/0, /*popm=*/0);

            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_tmp_id, dfb_max_scaler_id, dfb_max_id>(
                ckl::ReduceInputBlockShape::single());
        } else {
            ckl::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_COL,
                dfb_in0_id,
                dfb_max_scaler_id,
                dfb_max_id,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_dfb<dfb_in0_id, dfb_mask_id, dfb_tmp_id>(Ht - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, dfb_tmp_id, dfb_max_scaler_id, dfb_max_id>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(dfb_max_id, 1));  // iteration=1, reload from dfb_max_id
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
                ckl::BroadcastDim::Row,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                kDataFormatReconfig),
            ckl::output(dfb_x_m_max_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Ht));

        dfb_x_m_max_obj.wait_front(Ht);
#ifdef SOFTMAX
        constexpr bool is_softmax = true;
#else
        constexpr bool is_softmax = false;
#endif
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(Ht - 1),
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
                ckl::Dst::D0>{Ht - 1},
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
            ReduceDim::REDUCE_COL,
            dfb_exps_id,
            dfb_sum_scaler_id,
            dfb_recipsumexps_id,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::col(Ht),
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
            ReduceDim::REDUCE_COL,
            dfb_exps_id,
            dfb_sum_scaler_id,
            dfb_recipsumexps_id,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::col(Ht),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        dfb_x_m_max_obj.wait_front(Ht);
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
                ckl::BroadcastDim::Row,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                kDataFormatReconfig),
            ckl::output(dfb_out0_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Ht));
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
                ckl::BroadcastDim::Row,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                kDataFormatReconfig),
            ckl::output(dfb_out0_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(Ht));
#endif
        dfb_x_m_max_obj.pop_front(Ht);
    }
}
