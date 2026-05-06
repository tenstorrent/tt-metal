// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

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
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_exps = tt::CBIndex::c_24;
    constexpr auto cb_recipsumexps = tt::CBIndex::c_25;
    constexpr auto cb_add = tt::CBIndex::c_26;
    constexpr auto cb_max = tt::CBIndex::c_27;
    constexpr auto cb_tmp = tt::CBIndex::c_28;

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t dim_size = get_compile_time_arg_val(1);

    binary_op_init_common(cb_in0, cb_exps, cb_out0);

    for (uint32_t n = 0; n < N; ++n) {
        // find max
        for (uint32_t i = 0; i < dim_size; ++i) {
            if (i == 0) {
                moreh_copy_chain<cb_in0, cb_max, /*idx=*/0, /*pop=*/true>();
            } else {
                cb_wait_front(cb_in0, onetile);
                cb_wait_front(cb_max, onetile);

                tile_regs_acquire();

                copy_tile_init_with_dt(cb_in0);
                copy_tile(cb_in0, 0, dst0);

                copy_tile_init_with_dt(cb_max);
                copy_tile(cb_max, 0, dst1);

                binary_max_tile_init();
                binary_max_tile(dst0, dst1, dst0);
                tile_regs_commit();

                cb_pop_front(cb_max, onetile);
                cb_reserve_back(cb_max, onetile);

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_max);
                tile_regs_release();

                cb_push_back(cb_max, onetile);
                cb_pop_front(cb_in0, onetile);
            }
        }

        // compute exp(x - max(x))
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef SOFTMAX
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_in0,
                cb_max,
                cb_tmp,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

            exp_tile_to_cb(cb_tmp, cb_exps);
#else
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_in0,
                cb_max,
                cb_tmp,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

            rexp_tile_to_cb(cb_tmp, cb_exps);
#endif

            if (i == 0) {
                moreh_copy_chain<cb_exps, cb_add, /*idx=*/0, /*pop=*/true>();
            } else {
                moreh_bin_chain<
                    compute_kernel_lib::BinaryFpuOp::Add,
                    cb_add,
                    cb_exps,
                    cb_add,
                    /*idxA=*/0,
                    /*idxB=*/0,
                    /*popA=*/true,
                    /*popB=*/true>();
            }
        }

#ifdef LOG
        // compute log(sum)
        log_tile_to_cb(cb_add, cb_recipsumexps);
#else
        // compute 1/sum(exp(x))
        recip_tile_to_cb(cb_add, cb_recipsumexps);
#endif

        // step 3, compute final result
        cb_wait_front(cb_recipsumexps, onetile);
        for (uint32_t i = 0; i < dim_size; ++i) {
#ifdef LOG
#ifdef SOFTMAX
            // x - max - log(sum)
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_in0,
                cb_max,
                cb_tmp,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_tmp,
                cb_recipsumexps,
                cb_out0,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();
#else
            // -x + max - log(sum)
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_max,
                cb_in0,
                cb_tmp,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/false,
                /*popB=*/true>();

            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_tmp,
                cb_recipsumexps,
                cb_out0,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();
#endif
#else
#ifdef SOFTMAX
            // exp(x - max) / sum
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_in0,
                cb_max,
                cb_tmp,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

            exp_tile_to_cb(cb_tmp, cb_exps);

            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Mul,
                cb_exps,
                cb_recipsumexps,
                cb_out0,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();
#else
            // rexp(x - max) / sum
            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Sub,
                cb_in0,
                cb_max,
                cb_tmp,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();

            rexp_tile_to_cb(cb_tmp, cb_exps);

            moreh_bin_chain<
                compute_kernel_lib::BinaryFpuOp::Mul,
                cb_exps,
                cb_recipsumexps,
                cb_out0,
                /*idxA=*/0,
                /*idxB=*/0,
                /*popA=*/true,
                /*popB=*/false>();
#endif
#endif
        }

        cb_pop_front(cb_recipsumexps, onetile);
        cb_pop_front(cb_max, onetile);
    }
}
