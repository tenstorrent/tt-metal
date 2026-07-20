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
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t step = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_param_in = tt::CBIndex::c_0;
    CircularBuffer cb_param_in_obj(cb_param_in);
    constexpr auto cb_grad_in = tt::CBIndex::c_1;
    CircularBuffer cb_grad_in_obj(cb_grad_in);
    constexpr auto cb_exp_avg_in = tt::CBIndex::c_2;
    CircularBuffer cb_exp_avg_in_obj(cb_exp_avg_in);
    constexpr auto cb_exp_avg_sq_in = tt::CBIndex::c_3;
    CircularBuffer cb_exp_avg_sq_in_obj(cb_exp_avg_sq_in);
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CBIndex::c_4;
    CircularBuffer cb_max_exp_avg_sq_in_obj(cb_max_exp_avg_sq_in);
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto cb_scalar_args = tt::CBIndex::c_5;
    CircularBuffer cb_scalar_args_obj(cb_scalar_args);
    constexpr auto cb_one = tt::CBIndex::c_6;
    CircularBuffer cb_one_obj(cb_one);
    constexpr auto cb_param_out = tt::CBIndex::c_16;
    constexpr auto cb_exp_avg_out = tt::CBIndex::c_17;
    constexpr auto cb_exp_avg_sq_out = tt::CBIndex::c_18;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CBIndex::c_19;
#endif

    constexpr auto tmp_cb_param = tt::CBIndex::c_24;
    constexpr auto tmp_cb_exp_avg = tt::CBIndex::c_25;
    constexpr auto tmp_cb_exp_avg_sq = tt::CBIndex::c_26;
    CircularBuffer tmp_cb_exp_avg_sq_obj(tmp_cb_exp_avg_sq);
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CBIndex::c_27;
    CircularBuffer tmp_cb_max_exp_avg_sq_obj(tmp_cb_max_exp_avg_sq);
#endif
    constexpr auto cb_beta1_exponent = tt::CBIndex::c_28;
    CircularBuffer cb_beta1_exponent_obj(cb_beta1_exponent);
    constexpr auto cb_beta2_exponent = tt::CBIndex::c_29;
    CircularBuffer cb_beta2_exponent_obj(cb_beta2_exponent);
    constexpr auto cb_tmp1 = tt::CBIndex::c_30;
    CircularBuffer cb_tmp1_obj(cb_tmp1);
    constexpr auto cb_tmp2 = tt::CBIndex::c_31;
    CircularBuffer cb_tmp2_obj(cb_tmp2);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

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
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_scalar_args,
                cb_param_in,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::input(
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{weight_decay_tile, 0u},
            ckl::PackTile<cb_tmp1>{});

        ckl::mul<
            cb_scalar_args,
            cb_tmp1,
            cb_tmp1,
            ckl::BroadcastDim::None,
            ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

        ckl::sub<
            cb_param_in,
            cb_tmp1,
            tmp_cb_param,
            ckl::BroadcastDim::None,
            ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_one,
                cb_scalar_args,
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::input(
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set)>{0u, beta1_tile},
            ckl::PackTile<cb_tmp1>{});

        ckl::mul<cb_grad_in, cb_tmp1, cb_tmp1, ckl::BroadcastDim::None, ckl::input(ckl::InputLifecycle::CallerManaged)>(
            ckl::EltwiseShape::tiles(onetile));

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_exp_avg_in,
                cb_scalar_args,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::input(
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set)>{0u, beta1_tile},
            ckl::PackTile<tmp_cb_exp_avg>{});

        ckl::add<tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg>(ckl::EltwiseShape::tiles(onetile));

        ckl::copy<tmp_cb_exp_avg, cb_exp_avg_out, ckl::input(ckl::InputLifecycle::HeldStream)>(
            ckl::EltwiseShape::tiles(onetile));
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_one,
                cb_scalar_args,
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::input(
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set)>{0u, beta2_tile},
            ckl::PackTile<cb_tmp1>{});

        ckl::square<cb_grad_in, cb_tmp2, ckl::input(ckl::InputLifecycle::CallerManaged)>(
            ckl::EltwiseShape::tiles(onetile));

        ckl::mul<cb_tmp1, cb_tmp2, cb_tmp1>(ckl::EltwiseShape::tiles(onetile));

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_exp_avg_sq_in,
                cb_scalar_args,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::input(
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set)>{0u, beta2_tile},
            ckl::PackTile<tmp_cb_exp_avg_sq>{});

        ckl::add<tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq>(ckl::EltwiseShape::tiles(onetile));

        ckl::copy<tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, ckl::input(ckl::InputLifecycle::HeldStream)>(
            ckl::EltwiseShape::tiles(onetile));
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_beta2_exponent = pow(beta2, step); Calculated from host

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_one,
                cb_beta2_exponent,
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_tmp1>{});

#ifdef AMSGRAD
        ckl::binary_sfpu<
            ckl::BinaryMax<>,
            cb_max_exp_avg_sq_in,
            tmp_cb_exp_avg_sq,
            tmp_cb_max_exp_avg_sq,
            ckl::input(ckl::InputLifecycle::CallerManaged),
            ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

        ckl::copy<tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out, ckl::input(ckl::InputLifecycle::HeldStream)>(
            ckl::EltwiseShape::tiles(onetile));
#endif

#ifdef AMSGRAD
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                tmp_cb_max_exp_avg_sq,
                cb_tmp1,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Sqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<cb_tmp1>{});
        tmp_cb_max_exp_avg_sq_obj.pop_front(onetile);
#else
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                tmp_cb_exp_avg_sq,
                cb_tmp1,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Sqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<cb_tmp1>{});
#endif
        tmp_cb_exp_avg_sq_obj.pop_front(onetile);

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_tmp1,
                cb_scalar_args,
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None,
                ckl::input(),
                ckl::input(
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set)>{0u, eps_tile},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_tmp1>{});

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_beta1_exponent = pow(beta1, step); Calculated from host

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_one,
                cb_beta1_exponent,
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None,
                ckl::input(ckl::InputLifecycle::CallerManaged),
                ckl::input(ckl::InputLifecycle::CallerManaged)>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_tmp2>{});

        ckl::mul<
            cb_scalar_args,
            cb_tmp2,
            cb_tmp2,
            ckl::BroadcastDim::None,
            ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));

        ckl::mul<cb_tmp2, tmp_cb_exp_avg, cb_tmp2>(ckl::EltwiseShape::tiles(onetile));

        ckl::mul<cb_tmp1, cb_tmp2, cb_tmp1>(ckl::EltwiseShape::tiles(onetile));

        ckl::sub<tmp_cb_param, cb_tmp1, cb_param_out>(ckl::EltwiseShape::tiles(onetile));

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
