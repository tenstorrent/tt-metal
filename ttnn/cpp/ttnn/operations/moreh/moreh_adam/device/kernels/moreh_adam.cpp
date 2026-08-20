// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/minmax.hpp"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

#ifdef FP32_DEST_ACC_EN
#define WITH_FP32_DEST_ACC(x) x
#else
#define WITH_FP32_DEST_ACC(x)
#endif

void kernel_main() {
    uint32_t step = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto dfb_param_in_id = tt::CBIndex::c_0;
    constexpr auto dfb_grad_in_id = tt::CBIndex::c_1;
    constexpr auto dfb_exp_avg_in_id = tt::CBIndex::c_2;
    constexpr auto dfb_exp_avg_sq_in_id = tt::CBIndex::c_3;
#ifdef AMSGRAD
    constexpr auto dfb_max_exp_avg_sq_in_id = tt::CBIndex::c_4;
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto dfb_scalar_args_id = tt::CBIndex::c_5;
    constexpr auto dfb_one_id = tt::CBIndex::c_6;
    constexpr auto dfb_param_out_id = tt::CBIndex::c_16;
    constexpr auto dfb_exp_avg_out_id = tt::CBIndex::c_17;
    constexpr auto dfb_exp_avg_sq_out_id = tt::CBIndex::c_18;
#ifdef AMSGRAD
    constexpr auto dfb_max_exp_avg_sq_out_id = tt::CBIndex::c_19;
#endif

    constexpr auto tmp_dfb_grad_id = tt::CBIndex::c_24;
    constexpr auto tmp_dfb_exp_avg_id = tt::CBIndex::c_25;
    constexpr auto tmp_dfb_exp_avg_sq_id = tt::CBIndex::c_26;
#ifdef AMSGRAD
    constexpr auto tmp_dfb_max_exp_avg_sq_id = tt::CBIndex::c_27;
#endif
    constexpr auto dfb_tmp1_id = tt::CBIndex::c_30;
    constexpr auto dfb_tmp2_id = tt::CBIndex::c_31;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t beta1_tile = 1;
    constexpr uint32_t beta2_tile = 2;
    constexpr uint32_t eps_tile = 3;
    constexpr uint32_t weight_decay_tile = 4;
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_param_in_obj(dfb_param_in_id);
    DataflowBuffer dfb_grad_in_obj(dfb_grad_in_id);
    DataflowBuffer dfb_exp_avg_in_obj(dfb_exp_avg_in_id);
    DataflowBuffer dfb_exp_avg_sq_in_obj(dfb_exp_avg_sq_in_id);
#ifdef AMSGRAD
    DataflowBuffer dfb_max_exp_avg_sq_in_obj(dfb_max_exp_avg_sq_in_id);
#endif
    DataflowBuffer dfb_scalar_args_obj(dfb_scalar_args_id);
    DataflowBuffer dfb_one_obj(dfb_one_id);

    dfb_scalar_args_obj.wait_front(5);
    dfb_one_obj.wait_front(onetile);

    compute_kernel_hw_startup(dfb_param_in_id, dfb_scalar_args_id, dfb_param_out_id);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // grad += grad + param * weight_decay;
        // dfb_tmp1_id : param * weight_decay;
        dfb_param_in_obj.wait_front(onetile);
        dfb_grad_in_obj.wait_front(onetile);
        dfb_exp_avg_in_obj.wait_front(onetile);
        dfb_exp_avg_sq_in_obj.wait_front(onetile);
#ifdef AMSGRAD
        dfb_max_exp_avg_sq_in_obj.wait_front(onetile);
#endif
        // dfb_tmp1_id : param * weight_decay;
        mul_tiles_to_dfb<dfb_param_in_id, dfb_scalar_args_id, dfb_tmp1_id>(first_tile, weight_decay_tile, 0, 0);

        // tmp_dfb_grad_id : dfb_grad_in_id + dfb_tmp1_id;
        add_tiles_to_dfb<dfb_grad_in_id, dfb_tmp1_id, tmp_dfb_grad_id>(first_tile, first_tile, 0);

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // dfb_tmp1_id = (1 - beta1)
        sub_tiles_to_dfb<dfb_one_id, dfb_scalar_args_id, dfb_tmp1_id>(first_tile, beta1_tile, 0, 0);
        mul_tiles_to_dfb<tmp_dfb_grad_id, dfb_tmp1_id, dfb_tmp1_id>(first_tile, first_tile, 0);

        // tmp_dfb_exp_avg_id = dfb_exp_avg_in_id * beta1
        mul_tiles_to_dfb<dfb_exp_avg_in_id, dfb_scalar_args_id, tmp_dfb_exp_avg_id>(first_tile, beta1_tile, 0, 0);

        // tmp_dfb_exp_avg_id = tmp_dfb_exp_avg_id + dfb_tmp1_id
        add_tiles_to_dfb<tmp_dfb_exp_avg_id, dfb_tmp1_id, tmp_dfb_exp_avg_id>();

        // dfb_exp_avg_out_id
        copy_tile_to_dfb<tmp_dfb_exp_avg_id, dfb_exp_avg_out_id>(first_tile, 0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        sub_tiles_to_dfb<dfb_one_id, dfb_scalar_args_id, dfb_tmp1_id>(first_tile, beta2_tile, 0, 0);

        // dfb_tmp2_id = grad * grad
        mul_tiles_to_dfb<tmp_dfb_grad_id, tmp_dfb_grad_id, dfb_tmp2_id>(first_tile, first_tile, 1, 0);

        // dfb_tmp1_id = dfb_tmp1_id * dfb_tmp2_id
        mul_tiles_to_dfb<dfb_tmp1_id, dfb_tmp2_id, dfb_tmp1_id>();

        // tmp_dfb_exp_avg_sq_id = dfb_exp_avg_sq_in_id * beta2
        mul_tiles_to_dfb<dfb_exp_avg_sq_in_id, dfb_scalar_args_id, tmp_dfb_exp_avg_sq_id>(first_tile, beta2_tile, 0, 0);

        // tmp_dfb_exp_avg_sq_id = tmp_dfb_exp_avg_sq_id + dfb_tmp1_id
        add_tiles_to_dfb<tmp_dfb_exp_avg_sq_id, dfb_tmp1_id, tmp_dfb_exp_avg_sq_id>();

        // dfb_exp_avg_sq_out_id
        copy_tile_to_dfb<tmp_dfb_exp_avg_sq_id, dfb_exp_avg_sq_out_id>(first_tile, 0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // dfb_tmp1_id = pow(beta2, step);
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::CopyTile<
                ckl::input(
                    dfb_scalar_args_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset),
                ckl::Dst::D0>{beta2_tile},
            ckl::Power<ckl::Dst::D0>{step},
            ckl::PackTile<ckl::output(
                dfb_tmp1_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // dfb_tmp1_id = 1 / (1 - dfb_tmp1_id);
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Sub,
                ckl::input(dfb_one_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::input(dfb_tmp1_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_tmp1_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

#ifdef AMSGRAD
        // tmp_dfb_max_exp_avg_sq_id = max(dfb_max_exp_avg_sq_in_id, tmp_dfb_exp_avg_sq_id);
        ckl::binary_sfpu<
            ckl::BinaryMax<>,
            ckl::input(dfb_max_exp_avg_sq_in_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
            ckl::input(tmp_dfb_exp_avg_sq_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, kDataFormatReconfig),
            ckl::output(
                tmp_dfb_max_exp_avg_sq_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
            ckl::IterationShape::tiles(onetile));

        // dfb_max_exp_avg_sq_out_id
        copy_tile_to_dfb<tmp_dfb_max_exp_avg_sq_id, dfb_max_exp_avg_sq_out_id>(first_tile, 0);
#endif

        // dfb_tmp1_id = sqrt(exp_avg_sq / dfb_tmp1_id);
#ifdef AMSGRAD
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    tmp_dfb_max_exp_avg_sq_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_tmp1_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
            ckl::Sqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_tmp1_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});
#else
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(tmp_dfb_exp_avg_sq_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_tmp1_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
            ckl::Sqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_tmp1_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});
#endif

        // dfb_tmp1_id = 1 / (dfb_tmp1_id + eps)
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb_tmp1_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb_scalar_args_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset)>{0u, eps_tile},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_tmp1_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // bias_correction1 = 1 - pow(beta1, step);
        // dfb_tmp2_id = pow(beta1, step);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    dfb_scalar_args_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset),
                ckl::Dst::D0>{beta1_tile},
            ckl::Power<ckl::Dst::D0>{step},
            ckl::PackTile<ckl::output(
                dfb_tmp2_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // dfb_tmp2_id = 1 / (1 - dfb_tmp2_id);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Sub,
                ckl::input(
                    dfb_one_id,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset),
                ckl::input(dfb_tmp2_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                dfb_tmp2_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // dfb_tmp2_id = lr * dfb_tmp2_id;
        mul_tiles_to_dfb<dfb_scalar_args_id, dfb_tmp2_id, dfb_tmp2_id>(lr_tile, first_tile, 0);

        // dfb_tmp2_id = dfb_tmp2_id * tmp_dfb_exp_avg_id;
        mul_tiles_to_dfb<dfb_tmp2_id, tmp_dfb_exp_avg_id, dfb_tmp2_id>();

        // dfb_tmp1_id = dfb_tmp1_id * dfb_tmp2_id;
        mul_tiles_to_dfb<dfb_tmp1_id, dfb_tmp2_id, dfb_tmp1_id>();

        // param = param - dfb_tmp1_id;
        sub_tiles_to_dfb<dfb_param_in_id, dfb_tmp1_id, dfb_param_out_id>(first_tile, first_tile, 0);

        dfb_param_in_obj.pop_front(onetile);
        dfb_grad_in_obj.pop_front(onetile);
        dfb_exp_avg_in_obj.pop_front(onetile);
        dfb_exp_avg_sq_in_obj.pop_front(onetile);
#ifdef AMSGRAD
        dfb_max_exp_avg_sq_in_obj.pop_front(onetile);
#endif
    }

    dfb_scalar_args_obj.pop_front(5);
    dfb_one_obj.pop_front(onetile);
}
