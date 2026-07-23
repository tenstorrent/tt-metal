// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // sub
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Mask, Negative
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

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    cb_mask_obj.wait_front(onetile);
    cb_max_scaler_obj.wait_front(onetile);
    cb_sum_scaler_obj.wait_front(onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if constexpr (Wt == 1) {
            mask_tile_to_cb<cb_in0, cb_mask, cb_tmp>(0, 0, /*pop0=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            // Phase 1: reduce Wt-1 full tiles into cb_max via the helper.
            // cb_in0 holds all Wt tiles persistently for later steps, so use
            // WaitUpfrontNoPop — the helper waits for the slice it needs and never pops.
            ckl::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                cb_in0,
                cb_max_scaler,
                cb_max,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop>(ckl::ReduceInputBlockShape::row(Wt - 1));

            // Phase 2: mask the last tile (index Wt-1, no pop) and continue reducing
            // into cb_max via Accumulate. The accumulator and output are both cb_max:
            // the helper waits+pops the previous tile, then packs+pushes the new one.
            mask_tile_to_cb<cb_in0, cb_mask, cb_tmp>(Wt - 1, 0, /*pop0=*/0, /*popm=*/0);
            compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                compute_kernel_lib::ReduceInputBlockShape::row(1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::Accumulate::at(cb_max, /*iter=*/1));
        }

        ckl::sub<
            ckl::input(cb_in0, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
            ckl::input(cb_max, ckl::InputLifecycle::Bulk, kDataFormatReconfig),
            ckl::output(cb_x_m_max, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(Wt));

        cb_x_m_max_obj.wait_front(Wt);
#ifdef SOFTMAX
        constexpr bool is_softmax = true;
#else
        constexpr bool is_softmax = false;
#endif
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(Wt - 1),
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
                ckl::Dst::D0>{Wt - 1},
            ckl::OptionalChainElement<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::CopyTile<ckl::input(cb_mask, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig), ckl::Dst::D1>{},
            ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_exps, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

#ifdef LOG
        // log(sum) - pop tiles after reduce
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
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
            cb_exps,
            cb_sum_scaler,
            cb_recipsumexps,
            ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
            ckl::ReduceInputBlockShape::row(Wt),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        cb_x_m_max_obj.wait_front(Wt);
#ifdef LOG
        ckl::sub<
            ckl::input(cb_x_m_max, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block, kDataFormatReconfig),
            ckl::input(cb_recipsumexps, ckl::InputLifecycle::Bulk, kDataFormatReconfig),
            ckl::output(cb_out0, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(Wt));
#else
        ckl::mul<
            ckl::input(cb_exps, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, kDataFormatReconfig),
            ckl::input(cb_recipsumexps, ckl::InputLifecycle::Bulk, kDataFormatReconfig),
            ckl::output(cb_out0, ckl::OutputLifecycle::Bulk, kDataFormatReconfig),
            ckl::BroadcastDim::Col>(ckl::EltwiseShape::tiles(Wt));
#endif
        cb_x_m_max_obj.pop_front(Wt);
    }
}
