// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Negative
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    using D = ckl::Dst;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_output_grad = tt::CBIndex::c_0;
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;

    init_sfpu(cb_output_grad, cb_input_grad);

#if defined(DIVISOR)
    ckl::unary<ckl::Recip<D::D0>, cb_divisor, cb_tmp1, ckl::input(ckl::InputLifecycle::Bulk)>(
        ckl::EltwiseShape::single());

    cb_wait_front(cb_tmp1, 1);
    cb_wait_front(cb_output_grad, 1);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                cb_tmp_weight,
                cb_output_grad,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<cb_tmp2>{});

        compute_kernel_lib::mul<
            cb_tmp2,
            cb_tmp1,
            cb_input_grad,
            compute_kernel_lib::BroadcastDim::Scalar,
            compute_kernel_lib::input(),
            compute_kernel_lib::input(compute_kernel_lib::InputLifecycle::CallerManaged)>(
            compute_kernel_lib::EltwiseShape::single());
    }

    cb_pop_front(cb_output_grad, 1);
    cb_pop_front(cb_tmp1, 1);
#else
    cb_wait_front(cb_output_grad, 1);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                cb_tmp_weight,
                cb_output_grad,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar,
                ckl::input(),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<cb_input_grad>{});
    }

    cb_pop_front(cb_output_grad, 1);
#endif
}
