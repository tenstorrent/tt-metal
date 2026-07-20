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
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    CircularBuffer cb_in0_obj(cb_in0);
    constexpr auto cb_mask = tt::CBIndex::c_1;
    CircularBuffer cb_mask_obj(cb_mask);
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    CircularBuffer cb_max_scaler_obj(cb_max_scaler);
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    CircularBuffer cb_sum_scaler_obj(cb_sum_scaler);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer cb_out0_obj(cb_out0);
    constexpr auto cb_exps = tt::CBIndex::c_24;
    CircularBuffer cb_exps_obj(cb_exps);
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    CircularBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_max = tt::CBIndex::c_26;
    CircularBuffer cb_max_obj(cb_max);
    constexpr auto cb_x_m_max = tt::CBIndex::c_27;
    CircularBuffer cb_x_m_max_obj(cb_x_m_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;
    CircularBuffer cb_tmp_obj(cb_tmp);

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    cb_mask_obj.wait_front(onetile);
    cb_max_scaler_obj.wait_front(onetile);
    cb_sum_scaler_obj.wait_front(onetile);

    for (uint32_t n = 0; n < N; ++n) {
        // find max value
        if (Wt == 1) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<cb_in0, ckl::Dst::D0, ckl::input(ckl::InputLifecycle::HeldStream)>{},
                ckl::CopyTile<cb_mask, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::CallerManaged)>{},
                ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                ckl::PackTile<cb_tmp>{});

            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                ckl::ReduceInputBlockShape::single());
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
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::CopyTile<
                    cb_in0,
                    ckl::Dst::D0,
                    ckl::input(
                        ckl::InputLifecycle::HeldBulk,
                        ckl::OperandKind::Scalar,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileOffset::Set)>{Wt - 1},
                ckl::CopyTile<cb_mask, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::CallerManaged)>{},
                ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                ckl::PackTile<cb_tmp>{});
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_tmp, cb_max_scaler, cb_max>(
                ckl::ReduceInputBlockShape::row(1),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_max, /*iter=*/1));
        }

        ckl::sub<
            cb_in0,
            cb_max,
            cb_x_m_max,
            ckl::BroadcastDim::Col,
            ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
            ckl::input(ckl::InputLifecycle::Bulk),
            ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(Wt));

        cb_x_m_max_obj.wait_front(Wt);
#ifdef SOFTMAX
        constexpr bool is_softmax = true;
#else
        constexpr bool is_softmax = false;
#endif
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(Wt - 1),
            ckl::CopyTile<
                cb_x_m_max,
                ckl::Dst::D0,
                ckl::input(ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block)>{},
            ckl::OptionalChainElement<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<cb_exps>{});

        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::CopyTile<
                cb_x_m_max,
                ckl::Dst::D0,
                ckl::input(
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set)>{Wt - 1},
            ckl::OptionalChainElement<!is_softmax, ckl::Negative<ckl::Dst::D0>>{},
            ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::CopyTile<cb_mask, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
            ckl::PackTile<cb_exps>{});

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
            cb_x_m_max,
            cb_recipsumexps,
            cb_out0,
            ckl::BroadcastDim::Col,
            ckl::input(ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block),
            ckl::input(ckl::InputLifecycle::Bulk),
            ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(Wt));
#else
        ckl::mul<
            cb_exps,
            cb_recipsumexps,
            cb_out0,
            ckl::BroadcastDim::Col,
            ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
            ckl::input(ckl::InputLifecycle::Bulk),
            ckl::output(ckl::OutputLifecycle::Bulk)>(ckl::EltwiseShape::tiles(Wt));
#endif
        cb_x_m_max_obj.pop_front(Wt);
    }
}
