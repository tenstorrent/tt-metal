// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CBIndex::c_0;
    constexpr auto cb_dy = tt::CBIndex::c_1;
    constexpr auto cb_dx = tt::CBIndex::c_16;

    constexpr auto cb_ydy = tt::CBIndex::c_24;  // y * dy
    constexpr auto cb_sum = tt::CBIndex::c_25;
    CircularBuffer cb_sum_obj(cb_sum);
    constexpr auto cb_dy_m_sum = tt::CBIndex::c_26;  // dy - sum

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_dy, cb_y, cb_dx);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy) — accumulator over C-dim. No bcast since cb_dy and cb_sum are
        // both full tiles (no axis-mask either).
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                compute_kernel_lib::copy<
                    cb_dy,
                    cb_sum,
                    compute_kernel_lib::CopyTileReconfig::Input,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>(onetile);
            } else {
                compute_kernel_lib::add<cb_sum, cb_dy, cb_sum>(onetile);
            }
        }

        // Per-tile result: exp(y) * sum then dy - that.
        for (uint32_t i = 0; i < dim_size; ++i) {
            constexpr auto cb_exp = tt::CBIndex::c_24;
            compute_kernel_lib::unary<
                compute_kernel_lib::Exp<
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Approx::Exact,
                    compute_kernel_lib::Dst::D0>,
                cb_y,
                cb_exp,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>(onetile);

            // sum * exp(y) — cb_sum held outside, cb_exp streaming.
            constexpr auto cb_inter2 = tt::CBIndex::c_26;
            compute_kernel_lib::mul<
                cb_sum,
                cb_exp,
                cb_inter2,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::InputLifecycle::HeldStream>(onetile);

            // dy - sum * exp(y).
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_dy,
                    cb_inter2,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_dx,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        }
        cb_sum_obj.pop_front(onetile);
#else
        // compute sum(y * dy) over C-dim. No bcast.
        for (uint32_t i = 0; i < dim_size; ++i) {
            compute_kernel_lib::mul<cb_y, cb_dy, cb_ydy>(onetile);

            if (i == 0) {
                compute_kernel_lib::copy<
                    cb_ydy,
                    cb_sum,
                    compute_kernel_lib::CopyTileReconfig::Input,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>(onetile);
            } else {
                compute_kernel_lib::add<cb_sum, cb_ydy, cb_sum>(onetile);
            }
        }

        // Final result loop. cb_sum held outside.
        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_dy,
                    cb_sum,
                    compute_kernel_lib::BinaryFpuOp::Sub,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::HeldStream,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_dy_m_sum,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#ifdef SOFTMAX
            // (dy - sum) * y. cb_y held outside (InputLifecycle::CallerManaged).
            compute_kernel_lib::mul<cb_dy_m_sum, cb_y, cb_dx>(onetile);
#else
            // -(dy - sum) * y.
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    cb_dy_m_sum,
                    cb_y,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::Negative<compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_dx,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::Output>{});
#endif
        }
        cb_sum_obj.pop_front(onetile);
#endif
    }
}
