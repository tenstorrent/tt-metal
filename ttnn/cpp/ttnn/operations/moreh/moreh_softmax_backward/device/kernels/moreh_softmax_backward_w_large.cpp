// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Mask, Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CBIndex::c_0;
    constexpr auto cb_dy = tt::CBIndex::c_1;
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask = tt::CBIndex::c_3;
    constexpr auto cb_dx = tt::CBIndex::c_16;

    constexpr auto cb_ydy = tt::CBIndex::c_24;  // y * dy
    constexpr auto cb_sum = tt::CBIndex::c_25;
    CircularBuffer cb_sum_obj(cb_sum);
    constexpr auto cb_inter2 = tt::CBIndex::c_26;
    constexpr auto cb_add = tt::CBIndex::c_27;

    binary_op_init_common(cb_y, cb_bcast_scaler, cb_dx);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                if (w == 0) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(onetile),
                        ckl::CopyTile<cb_dy>{},
                        ckl::CopyTile<cb_mask, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::HeldStream)>{},
                        ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                        ckl::PackTile<cb_add>{});
                } else {
                    constexpr auto cb_inter0 = tt::CBIndex::c_24;
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(onetile),
                        ckl::CopyTile<cb_dy>{},
                        ckl::CopyTile<cb_mask, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::HeldStream)>{},
                        ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                        ckl::PackTile<cb_inter0>{});
                    ckl::add<cb_add, cb_inter0, cb_add>(ckl::EltwiseShape::tiles(onetile));
                }
            } else {
                if (w == 0) {
                    ckl::copy<cb_dy, cb_add>(ckl::EltwiseShape::tiles(onetile));
                } else {
                    ckl::add<cb_add, cb_dy, cb_add>(ckl::EltwiseShape::tiles(onetile));
                }
            }
        }

        ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_add, cb_bcast_scaler, cb_sum>(
            ckl::ReduceInputBlockShape::single());

        for (uint32_t w = 0; w < Wt; w += onetile) {
            constexpr auto cb_exp = tt::CBIndex::c_24;
            ckl::unary<ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>, cb_y, cb_exp>(
                ckl::EltwiseShape::tiles(onetile));
            ckl::mul<
                cb_exp,
                cb_sum,
                cb_inter2,
                ckl::BroadcastDim::Col,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::HeldStream)>(ckl::EltwiseShape::tiles(onetile));
            ckl::sub<cb_dy, cb_inter2, cb_dx>(ckl::EltwiseShape::tiles(onetile));
        }

        cb_sum_obj.pop_front(onetile);
#else
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(onetile),
                    ckl::BinaryFpu<cb_y, cb_dy, ckl::BinaryFpuOp::Mul>{},
                    ckl::CopyTile<cb_mask, ckl::Dst::D1, ckl::input(ckl::InputLifecycle::HeldStream)>{},
                    ckl::Mask<DataFormat::Float16_b, ckl::Dst::D0>{},
                    ckl::PackTile<cb_ydy>{});
            } else {
                ckl::mul<cb_y, cb_dy, cb_ydy>(ckl::EltwiseShape::tiles(onetile));
            }

            if (w == 0) {
                ckl::copy<cb_ydy, cb_add>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<cb_add, cb_ydy, cb_add>(ckl::EltwiseShape::tiles(onetile));
            }
        }

        ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_add, cb_bcast_scaler, cb_sum>(
            ckl::ReduceInputBlockShape::single());

        for (uint32_t w = 0; w < Wt; w += onetile) {
            ckl::sub<
                cb_dy,
                cb_sum,
                cb_inter2,
                ckl::BroadcastDim::Col,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::HeldStream)>(ckl::EltwiseShape::tiles(onetile));
#ifdef SOFTMAX
            ckl::mul<cb_y, cb_inter2, cb_dx>(ckl::EltwiseShape::tiles(onetile));
#else
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<cb_y, cb_inter2, ckl::BinaryFpuOp::Mul>{},
                ckl::Negative<ckl::Dst::D0>{},
                ckl::PackTile<cb_dx>{});
#endif
        }

        cb_sum_obj.pop_front(onetile);
#endif
    }
}
