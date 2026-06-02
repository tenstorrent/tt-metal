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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

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
        // cb_tmp1 : param * weight_decay;
        mul_tiles_to_cb(cb_param_in, cb_scalar_args, cb_tmp1, first_tile, weight_decay_tile, 0, 0);

        // tmp_cb_grad : cb_grad_in + cb_tmp1;
        add_tiles_to_cb(cb_grad_in, cb_tmp1, tmp_cb_grad, first_tile, first_tile, 0, 1);

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        sub_tiles_to_cb(cb_one, cb_scalar_args, cb_tmp1, first_tile, beta1_tile, 0, 0);
        mul_tiles_to_cb(tmp_cb_grad, cb_tmp1, cb_tmp1, first_tile, first_tile, 0, 1);

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        mul_tiles_to_cb(cb_exp_avg_in, cb_scalar_args, tmp_cb_exp_avg, first_tile, beta1_tile, 0, 0);

        // tmp_cb_exp_avg = tmp_cb_exp_avg + cb_tmp1
        add_tiles_to_cb(tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg, first_tile, first_tile, 1, 1);

        // cb_exp_avg_out
        copy_tile_to_cb(tmp_cb_exp_avg, cb_exp_avg_out, first_tile, 0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        sub_tiles_to_cb(cb_one, cb_scalar_args, cb_tmp1, first_tile, beta2_tile, 0, 0);

        // cb_tmp2 = grad * grad
        mul_tiles_to_cb(tmp_cb_grad, tmp_cb_grad, cb_tmp2, first_tile, first_tile, 1, 0);

        // cb_tmp1 = cb_tmp1 * cb_tmp2
        mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile, 1, 1);

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        mul_tiles_to_cb(cb_exp_avg_sq_in, cb_scalar_args, tmp_cb_exp_avg_sq, first_tile, beta2_tile, 0, 0);

        // tmp_cb_exp_avg_sq = tmp_cb_exp_avg_sq + cb_tmp1
        add_tiles_to_cb(tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq, first_tile, first_tile, 1, 1);

        // cb_exp_avg_sq_out
        copy_tile_to_cb(tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, first_tile, 0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_tmp1 = pow(beta2, step);
        // Reconfig: copy_tile_init_with_dt -> Input. pack_tile_with_dt -> Output.
        // cb_scalar_args InputLifecycle::CallerManaged + Scalar + compute_kernel_lib::TileOffset::Set.
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_scalar_args,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::TileOffset::Set>{beta2_tile},
            compute_kernel_lib::Power<compute_kernel_lib::Dst::D0>{step},
            compute_kernel_lib::PackTile<
                cb_tmp1,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        // cb_tmp1 = 1 / (1 - cb_tmp1)  — same-CB in/out on cb_tmp1.
        // Reconfig audit: `WITH_FP32_DEST_ACC(reconfig_data_format(cb_one, cb_tmp1))`
        //   is conditional but sub_tiles_init reconfigs srca/srcb unconditionally.
        //   pack_tile_with_dt does pack reconfig. -> Input + Output.
        // Lifecycles: cb_one InputLifecycle::CallerManaged + Scalar (held outside, popped at MAIN end).
        //   cb_tmp1 InputLifecycle::Streaming (wait+pop per call) on read; OutputLifecycle::Streaming (reserve+push)
        //   on write; chain handles the same-CB in/out cleanly.
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                cb_one,
                cb_tmp1,
                compute_kernel_lib::BinaryFpuOp::Sub,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::Recip<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_tmp1,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq)
        // Two-CB read into D0/D1 + SFPU binary_max + pack to D0.
        // Reconfig: copy_tile_init_with_dt reconfigs srca for each copy ->
        //   CopyTileReconfig::Input on both. pack_tile_with_dt -> PackTileReconfig::Output.
        // Lifecycles: cb_max_exp_avg_sq_in InputLifecycle::CallerManaged (pre-pushed by reader, pop at MAIN end).
        //   tmp_cb_exp_avg_sq InputLifecycle::CallerManaged (caller-managed lifecycle, pop later).
        //   tmp_cb_max_exp_avg_sq OutputLifecycle::Streaming.
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_max_exp_avg_sq_in,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::CopyTile<
                tmp_cb_exp_avg_sq,
                compute_kernel_lib::Dst::D1,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::
                BinaryMax<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                tmp_cb_max_exp_avg_sq,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        // cb_max_exp_avg_sq_out = tmp_cb_max_exp_avg_sq[first_tile]
        // Reconfig: copy_tile_init_with_dt -> Input. pack_tile_with_dt -> Output.
        // tmp_cb_max_exp_avg_sq waited here, popped by the next chain that reuses
        // the same tile -> InputLifecycle::CallerManaged + Scalar (first_tile == 0 -> default TileBase).
        tmp_cb_max_exp_avg_sq_obj.wait_front(onetile);
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                tmp_cb_max_exp_avg_sq,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input>{},
            compute_kernel_lib::PackTile<
                cb_max_exp_avg_sq_out,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
#endif

        // cb_tmp1 = sqrt(exp_avg_sq * cb_tmp1)  — same-CB in/out on cb_tmp1.
        // A operand chosen at compile time: AMSGRAD -> tmp_cb_max_exp_avg_sq, else
        //   -> tmp_cb_exp_avg_sq. Both held externally (waited earlier, popped after
        //   the chain) -> InputLifecycle::CallerManaged + Scalar.
        // cb_tmp1: InputLifecycle::Streaming on read + OutputLifecycle::Streaming on write (chain handles same-CB
        //   in/out the same as moreh_adam's other recip stages).
        // Reconfig: mul_tiles_init + WITH_FP32_DEST_ACC reconfig -> Input.
        //   pack_tile_with_dt -> Output.
#ifdef AMSGRAD
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                tmp_cb_max_exp_avg_sq,
                cb_tmp1,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::Sqrt<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_tmp1,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
        tmp_cb_max_exp_avg_sq_obj.pop_front(onetile);
#else
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                tmp_cb_exp_avg_sq,
                cb_tmp1,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::Sqrt<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_tmp1,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
#endif
        tmp_cb_exp_avg_sq_obj.pop_front(onetile);

        // cb_tmp1 = 1 / (cb_tmp1 + eps)  — same-CB in/out on cb_tmp1; eps held in
        // cb_scalar_args at index eps_tile.
        // Reconfig: add_tiles_init + WITH_FP32_DEST_ACC reconfig -> Input.
        //   pack_tile_with_dt -> Output.
        // Lifecycles: cb_tmp1 InputLifecycle::Streaming on read + OutputLifecycle::Streaming on write (same-CB).
        //   cb_scalar_args InputLifecycle::CallerManaged + Scalar + compute_kernel_lib::TileOffset::Set.
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                cb_tmp1,
                cb_scalar_args,
                compute_kernel_lib::BinaryFpuOp::Add,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::TileOffset::Unset,
                compute_kernel_lib::TileOffset::Set>{0u, eps_tile},
            compute_kernel_lib::Recip<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_tmp1,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_tmp2 = pow(beta1, step);
        // Reconfig: copy_tile_init_with_dt -> Input. pack_tile_with_dt -> Output.
        // cb_scalar_args InputLifecycle::CallerManaged + Scalar + compute_kernel_lib::TileOffset::Set.
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::CopyTile<
                cb_scalar_args,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::Input,
                compute_kernel_lib::TileOffset::Set>{beta1_tile},
            compute_kernel_lib::Power<compute_kernel_lib::Dst::D0>{step},
            compute_kernel_lib::PackTile<
                cb_tmp2,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        // cb_tmp2 = 1 / (1 - cb_tmp2)  — same-CB in/out on cb_tmp2.
        // Reconfig: sub_tiles_init + WITH_FP32_DEST_ACC reconfig -> Input.
        //   pack_tile_with_dt -> Output.
        // Lifecycles: cb_one InputLifecycle::CallerManaged + Scalar; cb_tmp2 InputLifecycle::Streaming on read +
        //   OutputLifecycle::Streaming on write.
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                cb_one,
                cb_tmp2,
                compute_kernel_lib::BinaryFpuOp::Sub,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::Recip<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_tmp2,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        // cb_tmp2 = lr * cb_tmp2;
        mul_tiles_to_cb(cb_scalar_args, cb_tmp2, cb_tmp2, lr_tile, first_tile, 0, 1);

        // cb_tmp2 = cb_tmp2 * tmp_cb_exp_avg;
        mul_tiles_to_cb(cb_tmp2, tmp_cb_exp_avg, cb_tmp2, first_tile, first_tile, 1, 1);

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile, 1, 1);

        // param = param - cb_tmp1;
        sub_tiles_to_cb(cb_param_in, cb_tmp1, cb_param_out, first_tile, first_tile, 0, 1);

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
