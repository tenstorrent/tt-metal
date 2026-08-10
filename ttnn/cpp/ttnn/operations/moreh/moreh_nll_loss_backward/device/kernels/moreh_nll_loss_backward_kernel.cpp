// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"       // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"       // Negative
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    using D = ckl::Dst;
    constexpr uint32_t dfb_divisor_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_output_grad_id = tt::CBIndex::c_0;
    DataflowBuffer dfb_output_grad_obj(dfb_output_grad_id);
    constexpr uint32_t dfb_tmp_weight_id = tt::CBIndex::c_24;
    constexpr uint32_t dfb_tmp1_id = tt::CBIndex::c_25;
    DataflowBuffer dfb_tmp1_obj(dfb_tmp1_id);
    constexpr uint32_t dfb_tmp2_id = tt::CBIndex::c_26;
    constexpr uint32_t dfb_input_grad_id = tt::CBIndex::c_16;

#if defined(DIVISOR)
    compute_kernel_hw_startup(dfb_divisor_id, dfb_tmp1_id);

    ckl::unary<
        ckl::Recip<D::D0>,
        ckl::input(dfb_divisor_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
        ckl::output(dfb_tmp1_id)>(ckl::IterationShape::one_tile());

    dfb_tmp1_obj.wait_front(1);
    dfb_output_grad_obj.wait_front(1);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb_tmp_weight_id),
                ckl::input(
                    dfb_output_grad_id, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<ckl::output(dfb_tmp2_id)>{});

        compute_kernel_lib::mul<
            compute_kernel_lib::input(dfb_tmp2_id),
            compute_kernel_lib::input(
                dfb_tmp1_id,
                compute_kernel_lib::BroadcastDim::Scalar,
                compute_kernel_lib::WaitPolicy::None,
                compute_kernel_lib::PopPolicy::None),
            compute_kernel_lib::output(dfb_input_grad_id)>(compute_kernel_lib::IterationShape::one_tile());
    }

    dfb_output_grad_obj.pop_front(1);
    dfb_tmp1_obj.pop_front(1);
#else
    compute_kernel_hw_startup(dfb_tmp_weight_id, dfb_output_grad_id, dfb_input_grad_id);

    dfb_output_grad_obj.wait_front(1);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb_tmp_weight_id),
                ckl::input(
                    dfb_output_grad_id, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<ckl::output(dfb_input_grad_id)>{});
    }

    dfb_output_grad_obj.pop_front(1);
#endif
}
