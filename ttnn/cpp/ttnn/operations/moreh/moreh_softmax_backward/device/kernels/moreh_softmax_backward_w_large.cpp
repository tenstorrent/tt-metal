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
        // sum(dy) — accumulate with mask on last tile.
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                if (w == 0) {
                    compute_kernel_lib::eltwise_chain(
                        onetile,
                        compute_kernel_lib::CopyTile<cb_dy>{},
                        compute_kernel_lib::CopyTile<
                            cb_mask,
                            compute_kernel_lib::Dst::D1,
                            compute_kernel_lib::InputLifecycle::HeldStream>{},
                        compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                        compute_kernel_lib::PackTile<cb_add>{});
                } else {
                    constexpr auto cb_inter0 = tt::CBIndex::c_24;
                    compute_kernel_lib::eltwise_chain(
                        onetile,
                        compute_kernel_lib::CopyTile<cb_dy>{},
                        compute_kernel_lib::CopyTile<
                            cb_mask,
                            compute_kernel_lib::Dst::D1,
                            compute_kernel_lib::InputLifecycle::HeldStream>{},
                        compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                        compute_kernel_lib::PackTile<cb_inter0>{});
                    compute_kernel_lib::add<cb_add, cb_inter0, cb_add>(onetile);
                }
            } else {
                if (w == 0) {
                    compute_kernel_lib::copy<cb_dy, cb_add>(onetile);
                } else {
                    compute_kernel_lib::add<cb_add, cb_dy, cb_add>(onetile);
                }
            }
        }

        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_add, cb_bcast_scaler, cb_sum, compute_kernel_lib::ReduceInputBlockShape::single());

        for (uint32_t w = 0; w < Wt; w += onetile) {
            constexpr auto cb_exp = tt::CBIndex::c_24;
            compute_kernel_lib::unary<
                compute_kernel_lib::Exp<
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>,
                cb_y,
                cb_exp>(onetile);
            compute_kernel_lib::mul<
                cb_exp,
                cb_sum,
                cb_inter2,
                compute_kernel_lib::BroadcastDim::Col,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::HeldStream>(onetile);
            compute_kernel_lib::sub<cb_dy, cb_inter2, cb_dx>(onetile);
        }

        cb_sum_obj.pop_front(onetile);
#else
        // step 1: y*dy + accumulator.
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<cb_y, cb_dy, compute_kernel_lib::BinaryFpuOp::Mul>{},
                    compute_kernel_lib::CopyTile<
                        cb_mask,
                        compute_kernel_lib::Dst::D1,
                        compute_kernel_lib::InputLifecycle::HeldStream>{},
                    compute_kernel_lib::Mask<DataFormat::Float16_b, compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<cb_ydy>{});
            } else {
                compute_kernel_lib::mul<cb_y, cb_dy, cb_ydy>(onetile);
            }

            if (w == 0) {
                compute_kernel_lib::copy<cb_ydy, cb_add>(onetile);
            } else {
                compute_kernel_lib::add<cb_add, cb_ydy, cb_add>(onetile);
            }
        }

        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_add, cb_bcast_scaler, cb_sum, compute_kernel_lib::ReduceInputBlockShape::single());

        // Per-tile result.
        for (uint32_t w = 0; w < Wt; w += onetile) {
            compute_kernel_lib::sub<
                cb_dy,
                cb_sum,
                cb_inter2,
                compute_kernel_lib::BroadcastDim::Col,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::HeldStream>(onetile);
#ifdef SOFTMAX
            compute_kernel_lib::mul<cb_y, cb_inter2, cb_dx>(onetile);
#else
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<cb_y, cb_inter2, compute_kernel_lib::BinaryFpuOp::Mul>{},
                compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<cb_dx>{});
#endif
        }

        cb_sum_obj.pop_front(onetile);
#endif
    }
}
