// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Negative
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    using D = ckl::Dst;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_output_grad = tt::CBIndex::c_0;
    DataflowBuffer dfb_output_grad_obj(cb_output_grad);
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    DataflowBuffer dfb_tmp1_obj(cb_tmp1);
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;

    init_sfpu(cb_output_grad, cb_input_grad);

#if defined(DIVISOR)
    ckl::unary<ckl::Recip<D::D0>, ckl::input(cb_divisor, ckl::InputLifecycle::Bulk), ckl::output(cb_tmp1)>(
        ckl::EltwiseShape::single());

    dfb_tmp1_obj.wait_front(1);
    dfb_output_grad_obj.wait_front(1);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(cb_tmp_weight),
                ckl::input(cb_output_grad, ckl::InputLifecycle::CallerManaged),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<ckl::output(cb_tmp2)>{});

        compute_kernel_lib::mul<
            compute_kernel_lib::input(cb_tmp2),
            compute_kernel_lib::input(cb_tmp1, compute_kernel_lib::InputLifecycle::CallerManaged),
            compute_kernel_lib::output(cb_input_grad),
            compute_kernel_lib::BroadcastDim::Scalar>(compute_kernel_lib::EltwiseShape::single());
    }

    dfb_output_grad_obj.pop_front(1);
    dfb_tmp1_obj.pop_front(1);
#else
    dfb_output_grad_obj.wait_front(1);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(cb_tmp_weight),
                ckl::input(cb_output_grad, ckl::InputLifecycle::CallerManaged),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Scalar>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<ckl::output(cb_input_grad)>{});
    }

    dfb_output_grad_obj.pop_front(1);
#endif
}
