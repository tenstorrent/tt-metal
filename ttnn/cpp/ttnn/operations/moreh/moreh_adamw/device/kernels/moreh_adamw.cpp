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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

namespace {

template <
    compute_kernel_lib::BinaryFpuOp Op,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    uint32_t IdxA,
    uint32_t IdxB,
    bool PopA,
    bool PopB>
ALWI void moreh_bin_chain() {
    using namespace compute_kernel_lib;
    using BinElt = BinaryFpu<
        CbA,
        CbB,
        Op,
        BroadcastDim::None,
        BinaryFpuOutputPolicy::PerTile,
        BinaryDataFormatReconfig::InputAndOutput,
        PopA ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        PopB ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        IdxA == 0 ? CbIndexMode::FirstTile : CbIndexMode::Pinned,
        IdxB == 0 ? CbIndexMode::FirstTile : CbIndexMode::Pinned,
        Dst::D0,
        0,
        0,
        0,
        CbOut>;
    BinElt elt{};
    elt.a_tile_idx = IdxA;
    elt.b_tile_idx = IdxB;
    eltwise_chain(1, elt, PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

template <uint32_t CbIn, uint32_t CbOut, uint32_t Idx, bool Pop>
ALWI void moreh_copy_chain() {
    using namespace compute_kernel_lib;
    using CopyElt = CopyTile<
        CbIn,
        Dst::D0,
        Pop ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        Idx == 0 ? CbIndexMode::FirstTile : CbIndexMode::Pinned,
        CopyTileReconfig::Input>;
    CopyElt elt{};
    elt.cb_tile_idx = Idx;
    eltwise_chain(
        1,
        elt,
        PackTile<
            CbOut,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::Output>{});
}

}  // namespace

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

    constexpr auto tmp_cb_param = tt::CBIndex::c_24;
    constexpr auto tmp_cb_exp_avg = tt::CBIndex::c_25;
    constexpr auto tmp_cb_exp_avg_sq = tt::CBIndex::c_26;
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CBIndex::c_27;
#endif
    constexpr auto cb_beta1_exponent = tt::CBIndex::c_28;
    constexpr auto cb_beta2_exponent = tt::CBIndex::c_29;
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

    cb_wait_front(cb_scalar_args, 5);
    cb_wait_front(cb_one, onetile);
    cb_wait_front(cb_beta1_exponent, onetile);
    cb_wait_front(cb_beta2_exponent, onetile);

    binary_op_init_common(cb_param_in, cb_scalar_args, cb_param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_wait_front(cb_param_in, onetile);
        cb_wait_front(cb_grad_in, onetile);
        cb_wait_front(cb_exp_avg_in, onetile);
        cb_wait_front(cb_exp_avg_sq_in, onetile);
#ifdef AMSGRAD
        cb_wait_front(cb_max_exp_avg_sq_in, onetile);
#endif
        // param = param - lr * weight_decay * param.
        // cb_tmp1 : weight_decay * cb_param_in
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_scalar_args,
            cb_param_in,
            cb_tmp1,
            weight_decay_tile,
            first_tile,
            /*popA=*/false,
            /*popB=*/false>();

        // cb_tmp1 : lr * cb_tmp1
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_scalar_args,
            cb_tmp1,
            cb_tmp1,
            lr_tile,
            first_tile,
            /*popA=*/false,
            /*popB=*/true>();

        // tmp_cb_param : cb_param_in - cb_tmp1
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Sub,
            cb_param_in,
            cb_tmp1,
            tmp_cb_param,
            first_tile,
            first_tile,
            /*popA=*/false,
            /*popB=*/true>();

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Sub,
            cb_one,
            cb_scalar_args,
            cb_tmp1,
            first_tile,
            beta1_tile,
            /*popA=*/false,
            /*popB=*/false>();

        // cb_tmp1 = cb_grad_in * cb_tmp1
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_grad_in,
            cb_tmp1,
            cb_tmp1,
            first_tile,
            first_tile,
            /*popA=*/false,
            /*popB=*/true>();

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_exp_avg_in,
            cb_scalar_args,
            tmp_cb_exp_avg,
            first_tile,
            beta1_tile,
            /*popA=*/false,
            /*popB=*/false>();

        // tmp_cb_exp_avg = tmp_cb_exp_avg + cb_tmp1
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Add,
            tmp_cb_exp_avg,
            cb_tmp1,
            tmp_cb_exp_avg,
            first_tile,
            first_tile,
            /*popA=*/true,
            /*popB=*/true>();

        // cb_exp_avg_out
        moreh_copy_chain<tmp_cb_exp_avg, cb_exp_avg_out, first_tile, /*pop=*/false>();
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        // cb_tmp1 = (1 - beta2)
        tile_regs_acquire();
        cb_reserve_back(cb_tmp1, onetile);
        sub_tiles_init_with_dt(cb_one, cb_scalar_args);
        sub_tiles(cb_one, cb_scalar_args, first_tile, beta2_tile, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_push_back(cb_tmp1, onetile);
        tile_regs_release();

        // cb_tmp2 = grad * grad
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_grad_in,
            cb_grad_in,
            cb_tmp2,
            first_tile,
            first_tile,
            /*popA=*/false,
            /*popB=*/false>();

        // cb_tmp1 = cb_tmp1 * cb_tmp2
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_tmp1,
            cb_tmp2,
            cb_tmp1,
            first_tile,
            first_tile,
            /*popA=*/true,
            /*popB=*/true>();

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_exp_avg_sq_in,
            cb_scalar_args,
            tmp_cb_exp_avg_sq,
            first_tile,
            beta2_tile,
            /*popA=*/false,
            /*popB=*/false>();

        // tmp_cb_exp_avg_sq = tmp_cb_exp_avg_sq + cb_tmp1
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Add,
            tmp_cb_exp_avg_sq,
            cb_tmp1,
            tmp_cb_exp_avg_sq,
            first_tile,
            first_tile,
            /*popA=*/true,
            /*popB=*/true>();

        // cb_exp_avg_sq_out
        moreh_copy_chain<tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, first_tile, /*pop=*/false>();
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_beta2_exponent = pow(beta2, step); Calculated from host

        // cb_tmp1 = 1 / (1 - cb_beta2_exponent);
        tile_regs_acquire();
        cb_reserve_back(cb_tmp1, onetile);
        sub_tiles_init_with_dt(cb_one, cb_beta2_exponent);
        sub_tiles(cb_one, cb_beta2_exponent, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_push_back(cb_tmp1, onetile);
        tile_regs_release();

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq);
        tile_regs_acquire();
        cb_reserve_back(tmp_cb_max_exp_avg_sq, onetile);
        copy_tile_init_with_dt(cb_max_exp_avg_sq_in);
        copy_tile(cb_max_exp_avg_sq_in, first_tile, dst0);
        copy_tile_init_with_dt(tmp_cb_exp_avg_sq);
        copy_tile(tmp_cb_exp_avg_sq, first_tile, dst1);
        binary_max_tile_init();
        binary_max_tile(dst0, dst1, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_cb_max_exp_avg_sq);
        cb_push_back(tmp_cb_max_exp_avg_sq, onetile);
        tile_regs_release();

        // cb_max_exp_avg_sq_out
        moreh_copy_chain<tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out, first_tile, /*pop=*/false>();
#endif

        // cb_tmp1 = sqrt(exp_avg_sq / cb_tmp1);
        tile_regs_acquire();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
#ifdef AMSGRAD
        mul_tiles_init_with_dt(tmp_cb_max_exp_avg_sq, cb_tmp1);
        mul_tiles(tmp_cb_max_exp_avg_sq, cb_tmp1, first_tile, first_tile, dst0);
#else
        mul_tiles_init_with_dt(tmp_cb_exp_avg_sq, cb_tmp1);
        mul_tiles(tmp_cb_exp_avg_sq, cb_tmp1, first_tile, first_tile, dst0);
#endif
        sqrt_tile_init();
        sqrt_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
#ifdef AMSGRAD
        cb_pop_front(tmp_cb_max_exp_avg_sq, onetile);
#endif
        cb_pop_front(tmp_cb_exp_avg_sq, onetile);
        tile_regs_release();

        // cb_tmp1 = 1 / (cb_tmp1 + eps)
        tile_regs_acquire();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        add_tiles_init_with_dt(cb_tmp1, cb_scalar_args);
        add_tiles(cb_tmp1, cb_scalar_args, first_tile, eps_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
        tile_regs_release();

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_beta1_exponent = pow(beta1, step); Calculated from host

        // cb_tmp2 = 1 / (1 - cb_beta1_exponent);
        tile_regs_acquire();
        cb_reserve_back(cb_tmp2, onetile);
        sub_tiles_init_with_dt(cb_one, cb_beta1_exponent);
        sub_tiles(cb_one, cb_beta1_exponent, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp2);
        cb_push_back(cb_tmp2, onetile);
        tile_regs_release();

        // cb_tmp2 = lr * cb_tmp2;
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_scalar_args,
            cb_tmp2,
            cb_tmp2,
            lr_tile,
            first_tile,
            /*popA=*/false,
            /*popB=*/true>();

        // cb_tmp2 = cb_tmp2 * tmp_cb_exp_avg;
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_tmp2,
            tmp_cb_exp_avg,
            cb_tmp2,
            first_tile,
            first_tile,
            /*popA=*/true,
            /*popB=*/true>();

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_tmp1,
            cb_tmp2,
            cb_tmp1,
            first_tile,
            first_tile,
            /*popA=*/true,
            /*popB=*/true>();

        // param = tmp_cb_param - cb_tmp1;
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Sub,
            tmp_cb_param,
            cb_tmp1,
            cb_param_out,
            first_tile,
            first_tile,
            /*popA=*/true,
            /*popB=*/true>();

        cb_pop_front(cb_param_in, onetile);
        cb_pop_front(cb_grad_in, onetile);
        cb_pop_front(cb_exp_avg_in, onetile);
        cb_pop_front(cb_exp_avg_sq_in, onetile);
#ifdef AMSGRAD
        cb_pop_front(cb_max_exp_avg_sq_in, onetile);
#endif
    }

    cb_pop_front(cb_scalar_args, 5);
    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_beta1_exponent, onetile);
    cb_pop_front(cb_beta2_exponent, onetile);
}
