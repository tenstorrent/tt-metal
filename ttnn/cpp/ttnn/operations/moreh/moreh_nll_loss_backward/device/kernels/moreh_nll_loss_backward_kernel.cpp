// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/special.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr auto per_core_tile_cnt = get_arg(args::per_core_tile_cnt);
    using D = ckl::Dst;

    DataflowBuffer dfb_output_grad_obj(dfb::output_grad);
#ifdef DIVISOR
    // These buffers are bound only for the divisor variant; keep their names out of the other variant.
    DataflowBuffer dfb_tmp1_obj(dfb::tmp1);

#if defined(FP32_DEST_ACC_EN)
    constexpr auto target_mask_data_format = DataFormat::Float32;
#else
    constexpr auto target_mask_data_format = DataFormat::Float16_b;
#endif

    compute_kernel_hw_startup(dfb::divisor, dfb::tmp1);
    ckl::unary<
        ckl::Recip<D::D0>,
        ckl::input(dfb::divisor, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckernel::moreh_data_format_reconfig),
        ckl::output(
            dfb::tmp1, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckernel::moreh_data_format_reconfig)>(
        ckl::IterationShape::one_tile());

    dfb_tmp1_obj.wait_front(1);
    dfb_output_grad_obj.wait_front(1);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    dfb::tmp_weight,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    ckernel::moreh_data_format_reconfig),
                ckl::input(
                    dfb::output_grad,
                    ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    ckernel::moreh_data_format_reconfig)>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<ckl::output(
                dfb::tmp2,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{});
        ckl::mul<
            ckl::input(
                dfb::tmp2, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckernel::moreh_data_format_reconfig),
            ckl::input(
                dfb::tmp1,
                ckl::BroadcastDim::Scalar,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Scalar,
                ckernel::moreh_data_format_reconfig),
            ckl::output(
                dfb::tmp3, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckernel::moreh_data_format_reconfig)>(
            ckl::IterationShape::one_tile());

        // `tmp_weight` is a dense one-hot/weight tile. When the divisor is zero, multiplying its
        // non-target zero lanes by the reciprocal produces NaN. Restore the sparse-write NLL
        // contract with an independent selected-target mask; a weight value of zero is still a
        // selected lane and must not be confused with an ignored target.
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    dfb::target_mask,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    ckernel::moreh_data_format_reconfig),
                D::D0>{},
            ckl::CopyTile<
                ckl::input(
                    dfb::tmp3, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckernel::moreh_data_format_reconfig),
                D::D1>{},
            ckl::FillBitcast<D::D2>{0U},
            ckl::Where<target_mask_data_format, D::D0, D::D1, D::D2, D::D0>{},
            ckl::PackTile<ckl::output(
                dfb::input_grad,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{});
    }
    dfb_output_grad_obj.pop_front(1);
    dfb_tmp1_obj.pop_front(1);
#else
    compute_kernel_hw_startup(dfb::tmp_weight, dfb::output_grad, dfb::input_grad);

    dfb_output_grad_obj.wait_front(1);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    dfb::tmp_weight,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    ckernel::moreh_data_format_reconfig),
                ckl::input(
                    dfb::output_grad,
                    ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    ckernel::moreh_data_format_reconfig)>{},
            ckl::Negative<D::D0>{},
            ckl::PackTile<ckl::output(
                dfb::input_grad,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckernel::moreh_data_format_reconfig)>{});
    }
    dfb_output_grad_obj.pop_front(1);
#endif
}
