// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    DataflowBuffer cb_mask_obj(cb_mask);
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    DataflowBuffer cb_max_scaler_obj(cb_max_scaler);
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    DataflowBuffer cb_sum_scaler_obj(cb_sum_scaler);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    constexpr auto cb_max = tt::CBIndex::c_26;
    constexpr auto cb_x_m_max = tt::CBIndex::c_27;
    DataflowBuffer cb_x_m_max_obj(cb_x_m_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(cb_in0, cb_max_scaler, cb_out0);

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);

    cb_mask_obj.wait_front(onetile);
    cb_max_scaler_obj.wait_front(onetile);
    cb_sum_scaler_obj.wait_front(onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if constexpr (Ht == 1) {
            mask_tile_to_cb<cb_in0, cb_mask, cb_tmp>(0, 0, /*pop0=*/0, /*popm=*/0);

            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, cb_tmp, cb_max_scaler, cb_max>(
                ckl::ReduceInputBlockShape::single());
        } else {
            ckl::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_COL,
                cb_in0,
                cb_max_scaler,
                cb_max,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                compute_kernel_lib::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_cb<cb_in0, cb_mask, cb_tmp>(Ht - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single(),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, 1));  // iteration=1, reload from cb_max
        }

        ckl::sub<
            ckl::input(cb_in0, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
            ckl::input(cb_max, ckl::InputLifecycle::Bulk, kDataFormatReconfig),
            ckl::output(cb_x_m_max, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
            ckl::BroadcastDim::Row>(ckl::EltwiseShape::tiles(Ht));

        cb_x_m_max_obj.wait_front(Ht);
#ifdef SOFTMAX
        constexpr bool is_softmax = true;
#else
        constexpr bool is_softmax = false;
#endif
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(Ht - 1),
            ckl::CopyTile<
                ckl::input(
                    cb_x_m_max, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block, kDataFormatReconfig),
                ckl::Dst::D0>{},
            ckl::OptionalChainElement<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_exps, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::CopyTile<
                ckl::input(
                    cb_x_m_max,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Block,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::Dst::D0>{Ht - 1},
            ckl::OptionalChainElement<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::CopyTile<ckl::input(cb_mask, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig), ckl::Dst::D1>{},
            ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_exps, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

#ifdef LOG
        // log(sum) - pop tiles after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
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
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::col(Ht),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        cb_x_m_max_obj.wait_front(Ht);
#ifdef LOG
        ckl::sub<
            ckl::input(cb_x_m_max, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block, kDataFormatReconfig),
            ckl::input(cb_recipsumexps, ckl::InputLifecycle::Bulk, kDataFormatReconfig),
            ckl::output(cb_out0, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
            ckl::BroadcastDim::Row>(ckl::EltwiseShape::tiles(Ht));
#else
        ckl::mul<
            ckl::input(cb_exps, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
            ckl::input(cb_recipsumexps, ckl::InputLifecycle::Bulk, kDataFormatReconfig),
            ckl::output(cb_out0, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
            ckl::BroadcastDim::Row>(ckl::EltwiseShape::tiles(Ht));
#endif
        cb_x_m_max_obj.pop_front(Ht);
    }
}
