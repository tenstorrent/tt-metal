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

namespace ckl = compute_kernel_lib;

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
                ckl::copy<cb_dy, cb_sum>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<cb_sum, cb_dy, cb_sum>(ckl::EltwiseShape::tiles(onetile));
            }
        }

        // Per-tile result: exp(y) * sum then dy - that.
        for (uint32_t i = 0; i < dim_size; ++i) {
            constexpr auto cb_exp = tt::CBIndex::c_24;
            ckl::unary<ckl::Exp<ckl::Approx::Exact, ckl::Approx::Exact, ckl::Dst::D0>, cb_y, cb_exp>(
                ckl::EltwiseShape::tiles(onetile));

            // sum * exp(y) — cb_sum held outside, cb_exp streaming.
            constexpr auto cb_inter2 = tt::CBIndex::c_26;
            ckl::mul<cb_sum, cb_exp, cb_inter2, ckl::BroadcastDim::None, ckl::InputLifecycle::HeldStream>(
                ckl::EltwiseShape::tiles(onetile));

            // dy - sum * exp(y).
            ckl::sub<cb_dy, cb_inter2, cb_dx>(ckl::EltwiseShape::tiles(onetile));
        }
        cb_sum_obj.pop_front(onetile);
#else
        // compute sum(y * dy) over C-dim. No bcast.
        for (uint32_t i = 0; i < dim_size; ++i) {
            ckl::mul<cb_y, cb_dy, cb_ydy>(ckl::EltwiseShape::tiles(onetile));

            if (i == 0) {
                ckl::copy<cb_ydy, cb_sum>(ckl::EltwiseShape::tiles(onetile));
            } else {
                ckl::add<cb_sum, cb_ydy, cb_sum>(ckl::EltwiseShape::tiles(onetile));
            }
        }

        // Final result loop. cb_sum held outside.
        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum.
            ckl::sub<
                cb_dy,
                cb_sum,
                cb_dy_m_sum,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::HeldStream>(ckl::EltwiseShape::tiles(onetile));
#ifdef SOFTMAX
            // (dy - sum) * y. cb_y held outside (InputLifecycle::CallerManaged).
            ckl::mul<cb_dy_m_sum, cb_y, cb_dx>(ckl::EltwiseShape::tiles(onetile));
#else
            // -(dy - sum) * y.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<cb_dy_m_sum, cb_y, ckl::BinaryFpuOp::Mul>{},
                ckl::Negative<ckl::Dst::D0>{},
                ckl::PackTile<cb_dx>{});
#endif
        }
        cb_sum_obj.pop_front(onetile);
#endif
    }
}
