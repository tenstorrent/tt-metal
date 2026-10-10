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

    constexpr auto cb_param_in = tt::CBIndex::c_0;
    constexpr auto cb_grad_in = tt::CBIndex::c_1;
    constexpr auto cb_exp_avg_in = tt::CBIndex::c_2;
    constexpr auto cb_exp_avg_sq_in = tt::CBIndex::c_3;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CBIndex::c_4;
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto cb_scalar_args = tt::CBIndex::c_5;
    constexpr auto cb_one = tt::CBIndex::c_6;
    constexpr auto cb_param_out = tt::CBIndex::c_16;
    constexpr auto cb_exp_avg_out = tt::CBIndex::c_17;
    constexpr auto cb_exp_avg_sq_out = tt::CBIndex::c_18;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CBIndex::c_19;
#endif

    constexpr auto tmp_cb_grad = tt::CBIndex::c_24;
    constexpr auto tmp_cb_exp_avg = tt::CBIndex::c_25;
    constexpr auto tmp_cb_exp_avg_sq = tt::CBIndex::c_26;
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CBIndex::c_27;
#endif
    constexpr auto cb_tmp1 = tt::CBIndex::c_30;
    constexpr auto cb_tmp2 = tt::CBIndex::c_31;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t beta1_tile = 1;
    constexpr uint32_t beta2_tile = 2;
    constexpr uint32_t eps_tile = 3;
    constexpr uint32_t weight_decay_tile = 4;
    constexpr uint32_t onetile = 1;

    constexpr auto scalar_args_input = ckl::input(
        cb_scalar_args,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Scalar,
        kDataFormatReconfig,
        ckl::TileAddressing::Offset);
    constexpr auto one_input = ckl::input(cb_one, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig);
    constexpr auto tmp1_input =
        ckl::input(cb_tmp1, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig);
    constexpr auto tmp1_output =
        ckl::output(cb_tmp1, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig);
    constexpr auto tmp2_input =
        ckl::input(cb_tmp2, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig);
    constexpr auto tmp2_output =
        ckl::output(cb_tmp2, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig);
    constexpr auto exp_avg_sq_input =
        ckl::input(tmp_cb_exp_avg_sq, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, kDataFormatReconfig);
#ifdef AMSGRAD
    constexpr auto max_exp_avg_sq_input =
        ckl::input(cb_max_exp_avg_sq_in, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig);
    constexpr auto max_exp_avg_sq_output =
        ckl::output(tmp_cb_max_exp_avg_sq, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig);
#endif

    DataflowBuffer dfb_param_in_obj(cb_param_in);
    DataflowBuffer dfb_grad_in_obj(cb_grad_in);
    DataflowBuffer dfb_exp_avg_in_obj(cb_exp_avg_in);
    DataflowBuffer dfb_exp_avg_sq_in_obj(cb_exp_avg_sq_in);
#ifdef AMSGRAD
    DataflowBuffer dfb_max_exp_avg_sq_in_obj(cb_max_exp_avg_sq_in);
#endif
    DataflowBuffer dfb_scalar_args_obj(cb_scalar_args);
    DataflowBuffer dfb_one_obj(cb_one);

    dfb_scalar_args_obj.wait_front(5);
    dfb_one_obj.wait_front(onetile);

    compute_kernel_hw_startup(cb_param_in, cb_scalar_args, cb_param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // grad += grad + param * weight_decay;
        // cb_tmp1 : param * weight_decay;
        dfb_param_in_obj.wait_front(onetile);
        dfb_grad_in_obj.wait_front(onetile);
        dfb_exp_avg_in_obj.wait_front(onetile);
        dfb_exp_avg_sq_in_obj.wait_front(onetile);
#ifdef AMSGRAD
        dfb_max_exp_avg_sq_in_obj.wait_front(onetile);
#endif
        // cb_tmp1 : param * weight_decay;
        mul_tiles_to_dfb<cb_param_in, cb_scalar_args, cb_tmp1>(first_tile, weight_decay_tile, 0, 0);

        // tmp_cb_grad : cb_grad_in + cb_tmp1;
        add_tiles_to_dfb<cb_grad_in, cb_tmp1, tmp_cb_grad>(first_tile, first_tile, 0);

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        sub_tiles_to_dfb<cb_one, cb_scalar_args, cb_tmp1>(first_tile, beta1_tile, 0, 0);
        mul_tiles_to_dfb<tmp_cb_grad, cb_tmp1, cb_tmp1>(first_tile, first_tile, 0);

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        mul_tiles_to_dfb<cb_exp_avg_in, cb_scalar_args, tmp_cb_exp_avg>(first_tile, beta1_tile, 0, 0);

        // tmp_cb_exp_avg = tmp_cb_exp_avg + cb_tmp1
        add_tiles_to_dfb<tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg>();

        // cb_exp_avg_out
        copy_tile_to_dfb<tmp_cb_exp_avg, cb_exp_avg_out>(first_tile, 0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        sub_tiles_to_dfb<cb_one, cb_scalar_args, cb_tmp1>(first_tile, beta2_tile, 0, 0);

        // cb_tmp2 = grad * grad
        mul_tiles_to_dfb<tmp_cb_grad, tmp_cb_grad, cb_tmp2>(first_tile, first_tile, 1, 0);

        // cb_tmp1 = cb_tmp1 * cb_tmp2
        mul_tiles_to_dfb<cb_tmp1, cb_tmp2, cb_tmp1>();

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        mul_tiles_to_dfb<cb_exp_avg_sq_in, cb_scalar_args, tmp_cb_exp_avg_sq>(first_tile, beta2_tile, 0, 0);

        // tmp_cb_exp_avg_sq = tmp_cb_exp_avg_sq + cb_tmp1
        add_tiles_to_dfb<tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq>();

        // cb_exp_avg_sq_out
        copy_tile_to_dfb<tmp_cb_exp_avg_sq, cb_exp_avg_sq_out>(first_tile, 0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_tmp1 = pow(beta2, step);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<scalar_args_input, ckl::Dst::D0>{beta2_tile},
            ckl::PowerIterative<ckl::Dst::D0>{step},
            ckl::PackTile<tmp1_output>{});

        // cb_tmp1 = 1 / (1 - cb_tmp1);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Sub, one_input, tmp1_input>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<tmp1_output>{});

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq);
        ckl::binary_sfpu<ckl::BinaryMax<>, max_exp_avg_sq_input, exp_avg_sq_input, max_exp_avg_sq_output>(
            ckl::IterationShape::one_tile());

        // cb_max_exp_avg_sq_out
        copy_tile_to_dfb<tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out>(first_tile, 0);
#endif

        // cb_tmp1 = sqrt(exp_avg_sq / cb_tmp1);
#ifdef AMSGRAD
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(tmp_cb_max_exp_avg_sq, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                tmp1_input>{},
            ckl::Sqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<tmp1_output>{});
#else
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, exp_avg_sq_input, tmp1_input>{},
            ckl::Sqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<tmp1_output>{});
#endif

        // cb_tmp1 = 1 / (cb_tmp1 + eps)
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Add, tmp1_input, scalar_args_input>{0u, eps_tile},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<tmp1_output>{});

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_tmp2 = pow(beta1, step);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<scalar_args_input, ckl::Dst::D0>{beta1_tile},
            ckl::PowerIterative<ckl::Dst::D0>{step},
            ckl::PackTile<tmp2_output>{});

        // cb_tmp2 = 1 / (1 - cb_tmp2);
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Sub, one_input, tmp2_input>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<tmp2_output>{});

        // cb_tmp2 = lr * cb_tmp2;
        mul_tiles_to_dfb<cb_scalar_args, cb_tmp2, cb_tmp2>(lr_tile, first_tile, 0);

        // cb_tmp2 = cb_tmp2 * tmp_cb_exp_avg;
        mul_tiles_to_dfb<cb_tmp2, tmp_cb_exp_avg, cb_tmp2>();

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        mul_tiles_to_dfb<cb_tmp1, cb_tmp2, cb_tmp1>();

        // param = param - cb_tmp1;
        sub_tiles_to_dfb<cb_param_in, cb_tmp1, cb_param_out>(first_tile, first_tile, 0);

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
