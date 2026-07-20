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
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_minmax.hpp"

namespace ckl = compute_kernel_lib;

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

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t beta1_tile = 1;
    constexpr uint32_t beta2_tile = 2;
    constexpr uint32_t eps_tile = 3;
    constexpr uint32_t weight_decay_tile = 4;
    constexpr uint32_t onetile = 1;

    CircularBuffer cb_param_in_obj(cb_param_in);
    CircularBuffer cb_grad_in_obj(cb_grad_in);
    CircularBuffer cb_exp_avg_in_obj(cb_exp_avg_in);
    CircularBuffer cb_exp_avg_sq_in_obj(cb_exp_avg_sq_in);
#ifdef AMSGRAD
    CircularBuffer cb_max_exp_avg_sq_in_obj(cb_max_exp_avg_sq_in);
    CircularBuffer tmp_cb_max_exp_avg_sq_obj(tmp_cb_max_exp_avg_sq);
    CircularBuffer cb_max_exp_avg_sq_out_obj(cb_max_exp_avg_sq_out);
#endif
    CircularBuffer cb_scalar_args_obj(cb_scalar_args);
    CircularBuffer cb_one_obj(cb_one);
    CircularBuffer cb_tmp1_obj(cb_tmp1);
    CircularBuffer cb_tmp2_obj(cb_tmp2);
    CircularBuffer tmp_cb_exp_avg_sq_obj(tmp_cb_exp_avg_sq);

    cb_scalar_args_obj.wait_front(5);
    cb_one_obj.wait_front(onetile);

    binary_op_init_common(cb_param_in, cb_scalar_args, cb_param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // grad += grad + param * weight_decay;
        // cb_tmp1 : param * weight_decay;
        cb_param_in_obj.wait_front(onetile);
        cb_grad_in_obj.wait_front(onetile);
        cb_exp_avg_in_obj.wait_front(onetile);
        cb_exp_avg_sq_in_obj.wait_front(onetile);
#ifdef AMSGRAD
        cb_max_exp_avg_sq_in_obj.wait_front(onetile);
#endif
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_param_in,
                cb_scalar_args,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::CallerManaged,
                ckl::InputLifecycle::CallerManaged,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Dst::D0,
                ckl::OperandKind::Scalar,
                ckl::OperandKind::Scalar,
                ckl::TileOffset::Unset,
                ckl::TileOffset::Set>{0u, weight_decay_tile},
            ckl::PackTile<cb_tmp1>{});

        ckl::add<
            cb_grad_in,
            cb_tmp1,
            tmp_cb_grad,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::CallerManaged,
            ckl::InputLifecycle::Streaming>(ckl::EltwiseShape::tiles(onetile));

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
                ckl::InputLifecycle::CallerManaged,
                ckl::InputLifecycle::CallerManaged,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Dst::D0,
                ckl::OperandKind::Scalar,
                ckl::OperandKind::Scalar,
                ckl::TileOffset::Unset,
                ckl::TileOffset::Set>{0u, beta1_tile},
            ckl::PackTile<cb_tmp1>{});
        ckl::mul<
            tmp_cb_grad,
            cb_tmp1,
            cb_tmp1,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::HeldStream,
            ckl::InputLifecycle::Streaming>(ckl::EltwiseShape::tiles(onetile));

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_exp_avg_in,
                cb_scalar_args,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::CallerManaged,
                ckl::InputLifecycle::CallerManaged,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Dst::D0,
                ckl::OperandKind::Scalar,
                ckl::OperandKind::Scalar,
                ckl::TileOffset::Unset,
                ckl::TileOffset::Set>{0u, beta1_tile},
            ckl::PackTile<tmp_cb_exp_avg>{});

        ckl::add<tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg>(ckl::EltwiseShape::tiles(onetile));

        ckl::copy<tmp_cb_exp_avg, cb_exp_avg_out, ckl::InputLifecycle::HeldStream>(ckl::EltwiseShape::tiles(onetile));
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
                ckl::InputLifecycle::CallerManaged,
                ckl::InputLifecycle::CallerManaged,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Dst::D0,
                ckl::OperandKind::Scalar,
                ckl::OperandKind::Scalar,
                ckl::TileOffset::Unset,
                ckl::TileOffset::Set>{0u, beta2_tile},
            ckl::PackTile<cb_tmp1>{});

        ckl::square<tmp_cb_grad, cb_tmp2>(ckl::EltwiseShape::tiles(onetile));

        ckl::mul<cb_tmp1, cb_tmp2, cb_tmp1>(ckl::EltwiseShape::tiles(onetile));

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_exp_avg_sq_in,
                cb_scalar_args,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::CallerManaged,
                ckl::InputLifecycle::CallerManaged,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Dst::D0,
                ckl::OperandKind::Scalar,
                ckl::OperandKind::Scalar,
                ckl::TileOffset::Unset,
                ckl::TileOffset::Set>{0u, beta2_tile},
            ckl::PackTile<tmp_cb_exp_avg_sq>{});

        ckl::add<tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq>(ckl::EltwiseShape::tiles(onetile));

        ckl::copy<tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, ckl::InputLifecycle::HeldStream>(
            ckl::EltwiseShape::tiles(onetile));
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_tmp1 = pow(beta2, step);
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<
                cb_scalar_args,
                ckl::Dst::D0,
                ckl::InputLifecycle::CallerManaged,
                ckl::CopyTileReconfig::Input,
                ckl::OperandKind::Scalar,
                ckl::TileOffset::Set>{beta2_tile},
            ckl::Power<ckl::Dst::D0>{step},
            ckl::PackTile<cb_tmp1>{});

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_one,
                cb_tmp1,
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::CallerManaged>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_tmp1>{});

#ifdef AMSGRAD
        ckl::binary_sfpu<
            ckl::BinaryMax<>,
            cb_max_exp_avg_sq_in,
            tmp_cb_exp_avg_sq,
            tmp_cb_max_exp_avg_sq,
            ckl::InputLifecycle::CallerManaged,
            ckl::InputLifecycle::CallerManaged>(ckl::EltwiseShape::tiles(onetile));

        tmp_cb_max_exp_avg_sq_obj.wait_front(onetile);
        ckl::copy<tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out, ckl::InputLifecycle::CallerManaged>(
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
                ckl::InputLifecycle::CallerManaged>{},
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
                ckl::InputLifecycle::CallerManaged>{},
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
                ckl::InputLifecycle::Streaming,
                ckl::InputLifecycle::CallerManaged,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Dst::D0,
                ckl::OperandKind::Scalar,
                ckl::OperandKind::Scalar,
                ckl::TileOffset::Unset,
                ckl::TileOffset::Set>{0u, eps_tile},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_tmp1>{});

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_tmp2 = pow(beta1, step);
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::CopyTile<
                cb_scalar_args,
                ckl::Dst::D0,
                ckl::InputLifecycle::CallerManaged,
                ckl::CopyTileReconfig::Input,
                ckl::OperandKind::Scalar,
                ckl::TileOffset::Set>{beta1_tile},
            ckl::Power<ckl::Dst::D0>{step},
            ckl::PackTile<cb_tmp2>{});

        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                cb_one,
                cb_tmp2,
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::None,
                ckl::InputLifecycle::CallerManaged>{},
            ckl::Recip<ckl::Dst::D0>{},
            ckl::PackTile<cb_tmp2>{});

        ckl::mul<
            cb_scalar_args,
            cb_tmp2,
            cb_tmp2,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::CallerManaged,
            ckl::InputLifecycle::Streaming>(ckl::EltwiseShape::tiles(onetile));

        ckl::mul<cb_tmp2, tmp_cb_exp_avg, cb_tmp2>(ckl::EltwiseShape::tiles(onetile));

        ckl::mul<cb_tmp1, cb_tmp2, cb_tmp1>(ckl::EltwiseShape::tiles(onetile));

        ckl::sub<
            cb_param_in,
            cb_tmp1,
            cb_param_out,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::CallerManaged,
            ckl::InputLifecycle::Streaming>(ckl::EltwiseShape::tiles(onetile));

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
}
