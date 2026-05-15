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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // BinaryMax
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Power, Recip, Sqrt
#include "ttnn/kernel/compute/moreh_common.hpp"

#ifdef FP32_DEST_ACC_EN
#define WITH_FP32_DEST_ACC(x) x
#else
#define WITH_FP32_DEST_ACC(x)
#endif

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
    // v6 Q4 collapse: conditional `IdxX == 0 ? FirstTile : Pinned` per-side has been
    // collapsed to uniform `Pinned`. Pinned-with-tile_idx=0 ≡ FirstTile semantically
    // (chain.inl exec() reads the per-side member field — default 0 propagates as 0).
    using BinElt = BinaryFpu<
        CbA,
        CbB,
        Op,
        BroadcastDim::None,
        BinaryDataFormatReconfig::Input,
        PopA ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        PopB ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        CbIndexMode::Pinned,
        Dst::D0>;
    eltwise_chain(
        1,
        BinElt{IdxA, IdxB},
        PackTile<
            CbOut,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::Output>{});
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
    eltwise_chain(
        1,
        CopyElt{Idx},
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

    cb_wait_front(cb_scalar_args, 5);
    cb_wait_front(cb_one, onetile);

    binary_op_init_common(cb_param_in, cb_scalar_args, cb_param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // grad += grad + param * weight_decay;
        // cb_tmp1 : param * weight_decay;
        cb_wait_front(cb_param_in, onetile);
        cb_wait_front(cb_grad_in, onetile);
        cb_wait_front(cb_exp_avg_in, onetile);
        cb_wait_front(cb_exp_avg_sq_in, onetile);
#ifdef AMSGRAD
        cb_wait_front(cb_max_exp_avg_sq_in, onetile);
#endif
        // cb_tmp1 : param * weight_decay;
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_param_in,
            cb_scalar_args,
            cb_tmp1,
            first_tile,
            weight_decay_tile,
            /*popA=*/false,
            /*popB=*/false>();

        // tmp_cb_grad : cb_grad_in + cb_tmp1;
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Add,
            cb_grad_in,
            cb_tmp1,
            tmp_cb_grad,
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
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            tmp_cb_grad,
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
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Sub,
            cb_one,
            cb_scalar_args,
            cb_tmp1,
            first_tile,
            beta2_tile,
            /*popA=*/false,
            /*popB=*/false>();

        // cb_tmp2 = grad * grad
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            tmp_cb_grad,
            tmp_cb_grad,
            cb_tmp2,
            first_tile,
            first_tile,
            /*popA=*/true,
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
        // cb_tmp1 = pow(beta2, step);  (T1.32)
        {
            using namespace compute_kernel_lib;
            eltwise_chain(
                onetile,
                CopyTile<
                    cb_scalar_args,
                    Dst::D0,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::Pinned,
                    CopyTileReconfig::Input>{beta2_tile},
                Power<Dst::D0>{step},
                PackTile<
                    cb_tmp1,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

        // cb_tmp1 = 1 / (1 - cb_tmp1);
        {
            using namespace compute_kernel_lib;
            eltwise_chain(
                onetile,
                BinaryFpu<
                    cb_one,
                    cb_tmp1,
                    BinaryFpuOp::Sub,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::NoWaitNoPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>{},
                Recip<Dst::D0>{},
                PackTile<
                    cb_tmp1,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq);
        {
            using namespace compute_kernel_lib;
            eltwise_chain(
                onetile,
                CopyTile<
                    cb_max_exp_avg_sq_in,
                    Dst::D0,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::FirstTile,
                    CopyTileReconfig::Input>{},
                CopyTile<
                    tmp_cb_exp_avg_sq,
                    Dst::D1,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::FirstTile,
                    CopyTileReconfig::Input>{},
                BinaryMax<Dst::D0, Dst::D1, Dst::D0>{},
                PackTile<
                    tmp_cb_max_exp_avg_sq,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

        // cb_max_exp_avg_sq_out  (T1.34)
        moreh_copy_chain<tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out, first_tile, /*pop=*/false>();
#endif

        // cb_tmp1 = sqrt(exp_avg_sq / cb_tmp1);
        {
            using namespace compute_kernel_lib;
#ifdef AMSGRAD
            eltwise_chain(
                onetile,
                BinaryFpu<
                    tmp_cb_max_exp_avg_sq,
                    cb_tmp1,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitNoPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>{},
                Sqrt<Approx::Exact, Dst::D0>{},
                PackTile<
                    cb_tmp1,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
#else
            eltwise_chain(
                onetile,
                BinaryFpu<
                    tmp_cb_exp_avg_sq,
                    cb_tmp1,
                    BinaryFpuOp::Mul,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitNoPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>{},
                Sqrt<Approx::Exact, Dst::D0>{},
                PackTile<
                    cb_tmp1,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
#endif
        }
#ifdef AMSGRAD
        cb_pop_front(tmp_cb_max_exp_avg_sq, onetile);
#endif
        cb_pop_front(tmp_cb_exp_avg_sq, onetile);

        // cb_tmp1 = 1 / (cb_tmp1 + eps)
        {
            using namespace compute_kernel_lib;
            eltwise_chain(
                onetile,
                BinaryFpu<
                    cb_tmp1,
                    cb_scalar_args,
                    BinaryFpuOp::Add,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::Pinned,
                    Dst::D0>{first_tile, eps_tile},
                Recip<Dst::D0>{},
                PackTile<
                    cb_tmp1,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_tmp2 = pow(beta1, step);  (T1.33)
        {
            using namespace compute_kernel_lib;
            eltwise_chain(
                onetile,
                CopyTile<
                    cb_scalar_args,
                    Dst::D0,
                    CopyTilePolicy::NoWaitNoPop,
                    CbIndexMode::Pinned,
                    CopyTileReconfig::Input>{beta1_tile},
                Power<Dst::D0>{step},
                PackTile<
                    cb_tmp2,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

        // cb_tmp2 = 1 / (1 - cb_tmp2);
        {
            using namespace compute_kernel_lib;
            eltwise_chain(
                onetile,
                BinaryFpu<
                    cb_one,
                    cb_tmp2,
                    BinaryFpuOp::Sub,
                    BroadcastDim::None,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::NoWaitNoPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    Dst::D0>{},
                Recip<Dst::D0>{},
                PackTile<
                    cb_tmp2,
                    Dst::D0,
                    PackTilePolicy::PerTileReserveAndPush,
                    PackTileIndexMode::FirstTile,
                    PackTileReconfig::Output>{});
        }

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

        // param = param - cb_tmp1;
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Sub,
            cb_param_in,
            cb_tmp1,
            cb_param_out,
            first_tile,
            first_tile,
            /*popA=*/false,
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
}
