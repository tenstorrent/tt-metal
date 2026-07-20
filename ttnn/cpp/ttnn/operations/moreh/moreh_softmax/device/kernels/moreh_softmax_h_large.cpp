// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp, Log, Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    CircularBuffer cb_recipsumexps_obj(cb_recipsumexps);
    constexpr auto cb_add = tt::CBIndex::c_26;
    constexpr auto cb_max = tt::CBIndex::c_27;
    CircularBuffer cb_max_obj(cb_max);
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    binary_op_init_common(cb_in0, cb_max_scaler, cb_out0);

    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
        // find max
        if (Ht == 1) {
            mask_tile_to_cb(
                DataflowBuffer(cb_in0), DataflowBuffer(cb_mask), DataflowBuffer(cb_tmp), 0, 0, /*pop0=*/1, /*popm=*/0);

            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, cb_tmp, cb_max_scaler, cb_max>(
                ckl::ReduceInputBlockShape::single());
        } else {
            // Phase 1: Reduce Ht-1 tiles
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, cb_in0, cb_max_scaler, cb_max>(
                ckl::ReduceInputBlockShape::col(Ht - 1));

            mask_tile_to_cb(
                DataflowBuffer(cb_in0), DataflowBuffer(cb_mask), DataflowBuffer(cb_tmp), 0, 0, /*pop0=*/1, /*popm=*/0);

            // Phase 2: Reduce final masked tile with accumulation
            ckl::reduce<PoolType::MAX, ReduceDim::REDUCE_COL, cb_tmp, cb_max_scaler, cb_max>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(cb_max, 1));
        }

        for (uint32_t h = 0; h < Ht; h += onetile) {
            // compute exp(x - max(x))
            if (h == Ht - 1) {
#ifdef SOFTMAX
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::BinaryFpu<
                        cb_in0,
                        cb_max,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::Row,
                        ckl::input(),
                        ckl::input(ckl::InputLifecycle::HeldStream)>{},
                    ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                    ckl::CopyTile<cb_mask, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::HeldStream)>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<cb_exps>{});
#else
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::CopyTile<cb_in0>{},
                    ckl::Negative<ckl::Dst::D0>{},
                    ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                    ckl::CopyTile<cb_mask, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::HeldStream)>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<cb_exps>{});
#endif
            } else {
#ifdef SOFTMAX
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::BinaryFpu<
                        cb_in0,
                        cb_max,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::Row,
                        ckl::input(),
                        ckl::input(ckl::InputLifecycle::HeldStream)>{},
                    ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                    ckl::PackTile<cb_exps>{});
#else
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::BinaryFpu<
                        cb_in0,
                        cb_max,
                        ckl::BinaryFpuOp::Sub,
                        ckl::BroadcastDim::Row,
                        ckl::input(),
                        ckl::input(ckl::InputLifecycle::HeldStream)>{},
                    ckl::Negative<ckl::Dst::D0>{},
                    ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                    ckl::PackTile<cb_exps>{});
#endif
            }

            if (h == 0) {
                ckl::copy<cb_exps, cb_add>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<cb_add, cb_exps, cb_add>(ckl::EltwiseShape::tiles(onetile));
            }
        }

#ifdef LOG
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_add,
            cb_sum_scaler,
            cb_recipsumexps,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::single(),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                log_tile_init();
                log_tile(dst_idx);
            });
#else
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            cb_add,
            cb_sum_scaler,
            cb_recipsumexps,
            ckl::ReduceInputPolicy::BulkWaitBulkPop>(
            ckl::ReduceInputBlockShape::single(),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });
#endif

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; h += onetile) {
#ifdef LOG
#ifdef SOFTMAX
            ckl::sub<
                cb_in0,
                cb_max,
                cb_tmp,
                ckl::BroadcastDim::Row,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::HeldStream)>(ckl::EltwiseShape::tiles(onetile));
            ckl::sub<
                cb_tmp,
                cb_recipsumexps,
                cb_out0,
                ckl::BroadcastDim::Row,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::HeldStream)>(ckl::EltwiseShape::tiles(onetile));
#else
#endif
#else
#ifdef SOFTMAX
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    cb_in0,
                    cb_max,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::Row,
                    ckl::input(),
                    ckl::input(ckl::InputLifecycle::HeldStream)>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<cb_exps>{});
            ckl::mul<
                cb_exps,
                cb_recipsumexps,
                cb_out0,
                ckl::BroadcastDim::Row,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::HeldStream)>(ckl::EltwiseShape::tiles(onetile));
#else
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    cb_in0,
                    cb_max,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::Row,
                    ckl::input(),
                    ckl::input(ckl::InputLifecycle::HeldStream)>{},
                ckl::Negative<ckl::Dst::D0>{},
                ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>{},
                ckl::PackTile<cb_exps>{});
            ckl::mul<
                cb_exps,
                cb_recipsumexps,
                cb_out0,
                ckl::BroadcastDim::Row,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::HeldStream)>(ckl::EltwiseShape::tiles(onetile));
#endif
#endif
        }

        cb_recipsumexps_obj.pop_front(onetile);
        cb_max_obj.pop_front(onetile);
    }
}
