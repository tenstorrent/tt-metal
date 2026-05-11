// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
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
        CbOut,
        Op,
        BroadcastDim::None,
        BinaryDataFormatReconfig::InputAndOutput,
        PopA ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        PopB ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        CbIndexMode::Pinned,
        Dst::D0,
        /*EnableFp32DestAcc=*/DST_ACCUM_MODE>;
    eltwise_chain(
        1,
        BinElt{IdxA, IdxB},
        PackTile<
            CbOut,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::Output,
            /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
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
            PackTileReconfig::Output,
            /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
}

// Unary SFPU chain: CopyTile(in, FirstTile, WaitAndPop) -> Sfpu(D0) -> PackTile(out).
template <typename Sfpu, uint32_t CbIn, uint32_t CbOut>
ALWI void moreh_unary_chain() {
    using namespace compute_kernel_lib;
    eltwise_chain(
        1,
        CopyTile<CbIn, Dst::D0, CopyTilePolicy::WaitAndPop, CbIndexMode::FirstTile, CopyTileReconfig::Input>{},
        Sfpu{},
        PackTile<
            CbOut,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::Output,
            /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
}

// BinaryFpu(Mul, None) + Negative + PackTile chain.
template <uint32_t CbA, uint32_t CbB, uint32_t CbOut, bool PopA, bool PopB>
ALWI void moreh_mul_neg_chain() {
    using namespace compute_kernel_lib;
    eltwise_chain(
        1,
        BinaryFpu<
            CbA,
            CbB,
            CbOut,
            BinaryFpuOp::Mul,
            BroadcastDim::None,
            BinaryDataFormatReconfig::InputAndOutput,
            PopA ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
            PopB ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
            CbIndexMode::FirstTile,
            Dst::D0,
            /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{},
        Negative<Dst::D0>{},
        PackTile<
            CbOut,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::Output,
            /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
}

}  // namespace

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CBIndex::c_0;
    constexpr auto cb_dy = tt::CBIndex::c_1;
    constexpr auto cb_dx = tt::CBIndex::c_16;

    constexpr auto cb_ydy = tt::CBIndex::c_24;  // y * dy
    constexpr auto cb_sum = tt::CBIndex::c_25;
    constexpr auto cb_dy_m_sum = tt::CBIndex::c_26;  // dy - sum

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_dy, cb_y, cb_dx);

    constexpr int dst0 = 0;
    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                moreh_copy_chain<cb_dy, cb_sum, /*idx=*/0, /*pop=*/true>();
            } else {
                moreh_bin_chain<
                    compute_kernel_lib::BinaryFpuOp::Add,
                    cb_sum,
                    cb_dy,
                    cb_sum,
                    /*idxA=*/0,
                    /*idxB=*/0,
                    /*popA=*/true,
                    /*popB=*/true>();
            }
        }

        for (uint32_t i = 0; i < dim_size; ++i) {
            // exp(y)  (T1.01)
            constexpr auto cb_exp = tt::CBIndex::c_24;
            moreh_unary_chain<compute_kernel_lib::Exp<>, cb_y, cb_exp>();

            // sum * exp(y)
            constexpr auto cb_inter2 = tt::CBIndex::c_26;
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Mul,
                cb_sum,
                cb_exp,
                cb_inter2,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/false,
                /*popB=*/true>();

            // dy - sum * exp(y)
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_dy,
                cb_inter2,
                cb_dx,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/true>();
        }
        cb_pop_front(cb_sum, onetile);
#else
        // compute sum(y * dy)
        for (uint32_t i = 0; i < dim_size; ++i) {
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Mul,
                cb_y,
                cb_dy,
                cb_ydy,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/true>();

            if (i == 0) {
                moreh_copy_chain<cb_ydy, cb_sum, /*idx=*/0, /*pop=*/true>();
            } else {
                moreh_bin_chain<
                    compute_kernel_lib::BinaryFpuOp::Add,
                    cb_sum,
                    cb_ydy,
                    cb_sum,
                    /*idxA=*/0,
                    /*idxB=*/0,
                    /*popA=*/true,
                    /*popB=*/true>();
            }
        }

        // compute final result
        for (uint32_t i = 0; i < dim_size; ++i) {
            // dy - sum
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_dy,
                cb_sum,
                cb_dy_m_sum,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

#ifdef SOFTMAX
            // (dy - sum) * y
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Mul,
                cb_dy_m_sum,
                cb_y,
                cb_dx,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/true>();
#else
            // -(dy - sum) * y  (T1.02)
            moreh_mul_neg_chain<cb_dy_m_sum, cb_y, cb_dx, /*PopA=*/true, /*PopB=*/true>();
#endif
        }
        cb_pop_front(cb_sum, onetile);
#endif
    }
}
