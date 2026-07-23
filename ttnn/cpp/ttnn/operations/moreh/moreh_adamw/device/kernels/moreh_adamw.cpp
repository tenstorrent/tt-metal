// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    uint32_t step = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_param_in = tt::CBIndex::c_0;
    DataflowBuffer cb_param_in_obj(cb_param_in);
    constexpr auto cb_grad_in = tt::CBIndex::c_1;
    DataflowBuffer cb_grad_in_obj(cb_grad_in);
    constexpr auto cb_exp_avg_in = tt::CBIndex::c_2;
    DataflowBuffer cb_exp_avg_in_obj(cb_exp_avg_in);
    constexpr auto cb_exp_avg_sq_in = tt::CBIndex::c_3;
    DataflowBuffer cb_exp_avg_sq_in_obj(cb_exp_avg_sq_in);
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CBIndex::c_4;
    DataflowBuffer cb_max_exp_avg_sq_in_obj(cb_max_exp_avg_sq_in);
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto cb_scalar_args = tt::CBIndex::c_5;
    DataflowBuffer cb_scalar_args_obj(cb_scalar_args);
    constexpr auto cb_one = tt::CBIndex::c_6;
    DataflowBuffer cb_one_obj(cb_one);
    constexpr auto cb_param_out = tt::CBIndex::c_16;
    constexpr auto cb_exp_avg_out = tt::CBIndex::c_17;
    constexpr auto cb_exp_avg_sq_out = tt::CBIndex::c_18;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CBIndex::c_19;
#endif

    constexpr auto tmp_cb_param = tt::CBIndex::c_24;
    constexpr auto tmp_cb_exp_avg = tt::CBIndex::c_25;
    constexpr auto tmp_cb_exp_avg_sq = tt::CBIndex::c_26;
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CBIndex::c_27;
#endif
    constexpr auto cb_beta1_exponent = tt::CBIndex::c_28;
    DataflowBuffer cb_beta1_exponent_obj(cb_beta1_exponent);
    constexpr auto cb_beta2_exponent = tt::CBIndex::c_29;
    DataflowBuffer cb_beta2_exponent_obj(cb_beta2_exponent);
    constexpr auto cb_tmp1 = tt::CBIndex::c_30;
    constexpr auto cb_tmp2 = tt::CBIndex::c_31;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t beta1_tile = 1;
    constexpr uint32_t beta2_tile = 2;
    constexpr uint32_t eps_tile = 3;
    constexpr uint32_t weight_decay_tile = 4;
    constexpr uint32_t onetile = 1;

    cb_scalar_args_obj.wait_front(5);
    cb_one_obj.wait_front(onetile);
    cb_beta1_exponent_obj.wait_front(onetile);
    cb_beta2_exponent_obj.wait_front(onetile);

    binary_op_init_common(cb_param_in, cb_scalar_args, cb_param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_param_in_obj.wait_front(onetile);
        cb_grad_in_obj.wait_front(onetile);
        cb_exp_avg_in_obj.wait_front(onetile);
        cb_exp_avg_sq_in_obj.wait_front(onetile);
#ifdef AMSGRAD
        cb_max_exp_avg_sq_in_obj.wait_front(onetile);
#endif
        // param = param - lr * weight_decay * param.
        // cb_tmp1 : weight_decay * cb_param_in
        mul_tiles_to_cb<cb_scalar_args, cb_param_in, cb_tmp1>(weight_decay_tile, first_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 : lr * cb_tmp1
        mul_tiles_to_cb<cb_scalar_args, cb_tmp1, cb_tmp1>(lr_tile, first_tile, /*pop0=*/0);

        // tmp_cb_param : cb_param_in - cb_tmp1
        sub_tiles_to_cb<cb_param_in, cb_tmp1, tmp_cb_param>(first_tile, first_tile, /*pop0=*/0);

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        sub_tiles_to_cb<cb_one, cb_scalar_args, cb_tmp1>(first_tile, beta1_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 = cb_grad_in * cb_tmp1
        mul_tiles_to_cb<cb_grad_in, cb_tmp1, cb_tmp1>(first_tile, first_tile, /*pop0=*/0);

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        mul_tiles_to_cb<cb_exp_avg_in, cb_scalar_args, tmp_cb_exp_avg>(
            first_tile,
            beta1_tile,
            /*pop0=*/0,
            /*pop1=*/0);

        // tmp_cb_exp_avg = tmp_cb_exp_avg + cb_tmp1
        add_tiles_to_cb<tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg>();

        // cb_exp_avg_out
        copy_tile_to_cb<tmp_cb_exp_avg, cb_exp_avg_out>(first_tile, /*pop=*/0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        // cb_tmp1 = (1 - beta2)
        sub_tiles_to_cb<cb_one, cb_scalar_args, cb_tmp1>(first_tile, beta2_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp2 = grad * grad
        mul_tiles_to_cb<cb_grad_in, cb_grad_in, cb_tmp2>(first_tile, first_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 = cb_tmp1 * cb_tmp2
        mul_tiles_to_cb<cb_tmp1, cb_tmp2, cb_tmp1>();

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        mul_tiles_to_cb<cb_exp_avg_sq_in, cb_scalar_args, tmp_cb_exp_avg_sq>(
            first_tile,
            beta2_tile,
            /*pop0=*/0,
            /*pop1=*/0);

        // tmp_cb_exp_avg_sq = tmp_cb_exp_avg_sq + cb_tmp1
        add_tiles_to_cb<tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq>();

        // cb_exp_avg_sq_out
        copy_tile_to_cb<tmp_cb_exp_avg_sq, cb_exp_avg_sq_out>(first_tile, /*pop=*/0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_beta2_exponent = pow(beta2, step); Calculated from host

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(cb_one, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::input(cb_beta2_exponent, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_tmp1, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq);
        ckl::binary_sfpu<
            ckl::BinaryMax<>,
            ckl::input(
                cb_max_exp_avg_sq_in,
                ckl::InputLifecycle::CallerManaged,
                ckl::OperandKind::Scalar,
                kDataFormatReconfig,
                ckl::TileOffset::Set),
            ckl::input(
                tmp_cb_exp_avg_sq, ckl::InputLifecycle::NoWaitPop, ckl::OperandKind::Scalar, kDataFormatReconfig),
            ckl::output(tmp_cb_max_exp_avg_sq, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>(
            ckl::EltwiseShape::single());

        // cb_max_exp_avg_sq_out
        copy_tile_to_cb<tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out>(first_tile, /*pop=*/0);
#endif

#ifdef AMSGRAD
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(tmp_cb_max_exp_avg_sq, ckl::InputLifecycle::NoWaitPop, kDataFormatReconfig),
                ckl::input(cb_tmp1, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None>{},
            ckl::Sqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_tmp1, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});
#else
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(tmp_cb_exp_avg_sq, ckl::InputLifecycle::NoWaitPop, kDataFormatReconfig),
                ckl::input(cb_tmp1, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None>{},
            ckl::Sqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_tmp1, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});
#endif

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(cb_tmp1, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(
                    cb_scalar_args,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None>{0u, eps_tile},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_tmp1, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_beta1_exponent = pow(beta1, step); Calculated from host

        // cb_tmp2 = 1 / (1 - cb_beta1_exponent);
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(
                    cb_one,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::input(
                    cb_beta1_exponent,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_tmp2, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

        // cb_tmp2 = lr * cb_tmp2;
        mul_tiles_to_cb<cb_scalar_args, cb_tmp2, cb_tmp2>(lr_tile, first_tile, /*pop0=*/0);

        // cb_tmp2 = cb_tmp2 * tmp_cb_exp_avg;
        mul_tiles_to_cb<cb_tmp2, tmp_cb_exp_avg, cb_tmp2>();

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        mul_tiles_to_cb<cb_tmp1, cb_tmp2, cb_tmp1>();

        // param = tmp_cb_param - cb_tmp1;
        sub_tiles_to_cb<tmp_cb_param, cb_tmp1, cb_param_out>();

        cb_param_in_obj.pop_front(onetile);
        cb_grad_in_obj.pop_front(onetile);
        cb_exp_avg_in_obj.pop_front(onetile);
        cb_exp_avg_sq_in_obj.pop_front(onetile);
#ifdef AMSGRAD
        cb_max_exp_avg_sq_in_obj.pop_front(onetile);
#endif
    }

    cb_scalar_args_obj.pop_front(5);
    cb_one_obj.pop_front(onetile);
    cb_beta1_exponent_obj.pop_front(onetile);
    cb_beta2_exponent_obj.pop_front(onetile);
}
