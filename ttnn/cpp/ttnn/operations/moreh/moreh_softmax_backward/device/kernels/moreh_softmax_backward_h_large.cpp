// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

namespace {

template <
    compute_kernel_lib::BinaryFpuOp Op,
    compute_kernel_lib::BroadcastDim Bcast,
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
        Bcast,
        BinaryDataFormatReconfig::InputAndOutput,
        PopA ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        PopB ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        CbIndexMode::Pinned,
        Dst::D0,
        /*EnableFp32DestAcc=*/DST_ACCUM_MODE>;
    BinElt elt{};
    elt.a_tile_idx = IdxA;
    elt.b_tile_idx = IdxB;
    eltwise_chain(
        1,
        elt,
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

// BinaryFpu(Mul, None) + Negative + PackTile chain (compile-time FirstTile indices).
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
    constexpr auto cb_bcast_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask = tt::CBIndex::c_3;
    constexpr auto cb_dx = tt::CBIndex::c_16;

    constexpr auto cb_ydy = tt::CBIndex::c_24;  // y * dy
    constexpr auto cb_sum = tt::CBIndex::c_25;
    constexpr auto cb_inter2 = tt::CBIndex::c_26;
    constexpr auto cb_add = tt::CBIndex::c_27;

    binary_op_init_common(cb_y, cb_bcast_scaler, cb_dx);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                if (h == 0) {
                    mask_tile_to_cb(cb_dy, cb_mask, cb_add, /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);
                } else {
                    constexpr auto cb_inter0 = tt::CBIndex::c_24;
                    mask_tile_to_cb(cb_dy, cb_mask, cb_inter0, /*itile=*/0, /*mtile=*/0, /*pop=*/1, /*popm=*/0);

                    moreh_bin_chain<
                        compute_kernel_lib::BinaryFpuOp::Add,
                        compute_kernel_lib::BroadcastDim::None,
                        cb_add,
                        cb_inter0,
                        cb_add,
                        /*idxA=*/0,
                        /*idxB=*/0,
                        /*popA=*/true,
                        /*popB=*/true>();
                }
            } else {
                if (h == 0) {
                    moreh_copy_chain<cb_dy, cb_add, /*idx=*/0, /*pop=*/true>();
                } else {
                    moreh_bin_chain<
                        compute_kernel_lib::BinaryFpuOp::Add,
                        compute_kernel_lib::BroadcastDim::None,
                        cb_add,
                        cb_dy,
                        cb_add,
                        /*idxA=*/0,
                        /*idxB=*/0,
                        /*popA=*/true,
                        /*popB=*/true>();
                }
            }
        }

        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL>(
            cb_add, cb_bcast_scaler, cb_sum, compute_kernel_lib::ReduceInputBlockShape::single());

        for (uint32_t h = 0; h < Ht; ++h) {
            // exp(y)  (T1.05)
            constexpr auto cb_exp = tt::CBIndex::c_24;
            moreh_unary_chain<compute_kernel_lib::Exp<>, cb_y, cb_exp>();

            // sum * exp(y)
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::Row,
                cb_exp,
                cb_sum,
                cb_inter2,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

            // dy - sum * exp(y)
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                compute_kernel_lib::BroadcastDim::None,
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

        // step 1, compute y * dy
        for (uint32_t h = 0; h < Ht; ++h) {
            if (h == Ht - 1) {
                mul_tiles_and_mask_tile_to_cb(
                    cb_y, cb_dy, cb_mask, cb_ydy, 0, 0, 0, /*pop0=*/1, /*pop1=*/1, /*popm=*/0);
            } else {
                moreh_bin_chain<
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    cb_y,
                    cb_dy,
                    cb_ydy,
                    /*idxA=*/0,
                    /*idxB=*/0,
                    /*popA=*/true,
                    /*popB=*/true>();
            }

            if (h == 0) {
                moreh_copy_chain<cb_ydy, cb_add, /*idx=*/0, /*pop=*/true>();
            } else {
                moreh_bin_chain<
                    compute_kernel_lib::BinaryFpuOp::Add,
                    compute_kernel_lib::BroadcastDim::None,
                    cb_add,
                    cb_ydy,
                    cb_add,
                    /*idxA=*/0,
                    /*idxB=*/0,
                    /*popA=*/true,
                    /*popB=*/true>();
            }
        }

        // step 2, compute sum(y * dy)
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL>(
            cb_add, cb_bcast_scaler, cb_sum, compute_kernel_lib::ReduceInputBlockShape::single());

        // step 3, compute final result
        for (uint32_t h = 0; h < Ht; ++h) {
            // dy - sum
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                compute_kernel_lib::BroadcastDim::Row,
                cb_dy,
                cb_sum,
                cb_inter2,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

#ifdef SOFTMAX
            // (dy - sum) * y
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                cb_y,
                cb_inter2,
                cb_dx,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/true>();
#else
            // -(dy - sum) * y  (T1.06)
            moreh_mul_neg_chain<cb_y, cb_inter2, cb_dx, /*PopA=*/true, /*PopB=*/true>();
#endif
        }

        cb_pop_front(cb_sum, onetile);
#endif
    }
}
