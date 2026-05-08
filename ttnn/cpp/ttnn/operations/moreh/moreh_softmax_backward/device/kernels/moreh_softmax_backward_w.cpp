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
        Dst::D0>;
    BinElt elt{};
    elt.a_tile_idx = IdxA;
    elt.b_tile_idx = IdxB;
    eltwise_chain(1, elt, PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

template <
    compute_kernel_lib::BinaryFpuOp Op,
    compute_kernel_lib::BroadcastDim Bcast,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    bool PopA,
    bool PopB>
ALWI void moreh_bin_chain_rt(uint32_t idxA, uint32_t idxB) {
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
        Dst::D0>;
    BinElt elt{};
    elt.a_tile_idx = idxA;
    elt.b_tile_idx = idxB;
    eltwise_chain(1, elt, PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

// Unary SFPU chain with runtime tile_idx (Pinned, WaitNoPop on input).
template <typename Sfpu, uint32_t CbIn, uint32_t CbOut>
ALWI void moreh_unary_chain_rt(uint32_t idx) {
    using namespace compute_kernel_lib;
    using CopyElt = CopyTile<CbIn, Dst::D0, CopyTilePolicy::WaitNoPop, CbIndexMode::Pinned>;
    CopyElt elt{};
    elt.cb_tile_idx = idx;
    eltwise_chain(1, elt, Sfpu{}, PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
}

// BinaryFpu(Mul, None, Pinned) + Negative + PackTile chain (runtime per-side tile_idx).
template <uint32_t CbA, uint32_t CbB, uint32_t CbOut, bool PopA, bool PopB>
ALWI void moreh_mul_neg_chain_rt(uint32_t idxA, uint32_t idxB) {
    using namespace compute_kernel_lib;
    using BinElt = BinaryFpu<
        CbA,
        CbB,
        CbOut,
        BinaryFpuOp::Mul,
        BroadcastDim::None,
        BinaryDataFormatReconfig::InputAndOutput,
        PopA ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        PopB ? CopyTilePolicy::WaitAndPop : CopyTilePolicy::WaitNoPop,
        CbIndexMode::Pinned,
        Dst::D0>;
    BinElt elt{};
    elt.a_tile_idx = idxA;
    elt.b_tile_idx = idxB;
    eltwise_chain(1, elt, Negative<Dst::D0>{}, PackTile<CbOut, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
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

    binary_op_init_common(cb_y, cb_bcast_scaler, cb_dx);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {
#ifdef LOG
        // sum(dy)
        if (Wt == 1) {
            // apply mask
            mask_tile_to_cb(cb_dy, cb_mask, cb_inter2, /*itile=*/0, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_inter2, cb_bcast_scaler, cb_sum, compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            constexpr auto cb_inter0 = tt::CBIndex::c_24;
            compute_kernel_lib::
                reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                    cb_dy, cb_bcast_scaler, cb_inter0, compute_kernel_lib::ReduceInputBlockShape::row(Wt - 1));

            constexpr auto cb_inter1 = tt::CBIndex::c_25;
            mask_tile_to_cb(cb_dy, cb_mask, cb_inter1, /*itile=*/Wt - 1, /*mtile=*/0, /*pop=*/0, /*popm=*/0);

            constexpr auto cb_inter2 = tt::CBIndex::c_26;
            compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
                cb_inter1, cb_bcast_scaler, cb_inter2, compute_kernel_lib::ReduceInputBlockShape::single());

            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Add,
                compute_kernel_lib::BroadcastDim::None,
                cb_inter0,
                cb_inter2,
                cb_sum,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/true>();
        }

        // dy - sum * exp(y)
        constexpr auto cb_exp = tt::CBIndex::c_24;  // y * dy

        for (uint32_t w = 0; w < Wt; w += onetile) {
            // exp(y)  (T1.07)
            moreh_unary_chain_rt<compute_kernel_lib::Exp<>, cb_y, cb_exp>(w);

            // sum * exp(y)
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::Col,
                cb_exp,
                cb_sum,
                cb_inter2,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

            // dy - sum * exp(y)
            moreh_bin_chain_rt<
                compute_kernel_lib::BinaryFpuOp::Sub,
                compute_kernel_lib::BroadcastDim::None,
                cb_dy,
                cb_inter2,
                cb_dx,
                /*popA=*/false,
                /*popB=*/true>(w, 0);
        }

        cb_pop_front(cb_sum, onetile);
        cb_pop_front(cb_y, Wt);
        cb_pop_front(cb_dy, Wt);
#else
        // step 1, compute y * dy
        for (uint32_t w = 0; w < Wt; ++w) {
            if (w == Wt - 1) {
                mul_tiles_and_mask_tile_to_cb(
                    cb_y, cb_dy, cb_mask, cb_ydy, w, w, 0, /*pop0=*/0, /*pop1=*/0, /*popm=*/0);
            } else {
                moreh_bin_chain_rt<
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    cb_y,
                    cb_dy,
                    cb_ydy,
                    /*popA=*/false,
                    /*popB=*/false>(w, w);
            }
        }

        // step 2, compute sum(y * dy)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_ydy, cb_bcast_scaler, cb_sum, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // step 3, compute final result
        for (uint32_t w = 0; w < Wt; w += onetile) {
            // dy - sum
            moreh_bin_chain_rt<
                compute_kernel_lib::BinaryFpuOp::Sub,
                compute_kernel_lib::BroadcastDim::Col,
                cb_dy,
                cb_sum,
                cb_inter2,
                /*popA=*/false,
                /*popB=*/false>(w, 0);

#ifdef SOFTMAX
            // (dy - sum) * y
            moreh_bin_chain_rt<
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                cb_y,
                cb_inter2,
                cb_dx,
                /*popA=*/false,
                /*popB=*/true>(w, 0);
#else
            // -(dy - sum) * y  (T1.08)
            moreh_mul_neg_chain_rt<cb_y, cb_inter2, cb_dx, /*PopA=*/false, /*PopB=*/true>(w, 0);
#endif
        }

        cb_pop_front(cb_sum, onetile);
        cb_pop_front(cb_dy, Wt);
        cb_pop_front(cb_y, Wt);
#endif
    }
}
