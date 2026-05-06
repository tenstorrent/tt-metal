// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

namespace {

// File-local helper: BinaryFpu(Op) on (CbA[idxA], CbB[idxB]) -> CbOut, single tile,
// PerTile policies with WaitAndPop / WaitNoPop driven by PopA / PopB. Mirrors the
// `*_tiles_to_cb` moreh helpers (FP32_DEST_ACC reconfig is folded into the chain via
// BinaryDataFormatReconfig::InputAndOutput — emits unconditionally, equivalent to the
// _with_dt wrappers under FP32_DEST_ACC and a no-op otherwise on bf16-only).
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
    constexpr auto cb_param_in = tt::CBIndex::c_0;
    constexpr auto cb_grad = tt::CBIndex::c_1;
    constexpr auto cb_momentum_in = tt::CBIndex::c_2;

    constexpr auto cb_param_out = tt::CBIndex::c_16;
    constexpr auto cb_momentum_out = tt::CBIndex::c_17;

    constexpr auto cb_scalar_args = tt::CBIndex::c_24;
    constexpr auto cb_tmp1 = tt::CBIndex::c_25;
    constexpr auto cb_tmp2 = tt::CBIndex::c_26;
    constexpr auto cb_tmp3 = tt::CBIndex::c_27;
    constexpr auto cb_tmp4 = tt::CBIndex::c_28;

    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t momentum_tile = 1;
    constexpr uint32_t dampening_tile = 2;
    constexpr uint32_t weight_decay_tile = 3;
    constexpr uint32_t one_tile = 4;

    binary_op_init_common(cb_param_in, cb_param_in, cb_param_out);

    uint32_t num_tiles = get_compile_time_arg_val(0);

    // from reader
    cb_wait_front(cb_scalar_args, 5);

    for (uint32_t n = 0; n < num_tiles; ++n) {
        uint32_t cb_grad_tmp = cb_grad;
#if defined(WEIGHT_DECAY)
        // grad += param * weight_decay
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_param_in,
            cb_scalar_args,
            cb_tmp1,
            /*idxA=*/0,
            /*idxB=*/weight_decay_tile,
            /*popA=*/false,
            /*popB=*/false>();

        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Add,
            cb_grad,
            cb_tmp1,
            cb_tmp2,
            /*idxA=*/0,
            /*idxB=*/0,
            /*popA=*/true,
            /*popB=*/true>();

        cb_grad_tmp = cb_tmp2;
#endif  // WEIGHT_DECAY

#if defined(MOMENTUM)
        uint32_t cb_momentum_tmp = cb_grad_tmp;
#if defined(MOMENTUM_INITIALIZED)
        // grad * (1 - dampening)
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Sub,
            cb_scalar_args,
            cb_scalar_args,
            cb_tmp1,
            /*idxA=*/one_tile,
            /*idxB=*/dampening_tile,
            /*popA=*/false,
            /*popB=*/false>();

        mul_tiles_to_cb(cb_grad_tmp, cb_tmp1, cb_tmp3, 0, 0, /*pop0=*/0, /*pop0=*/1);

        // momentum_v * momentum
        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Mul,
            cb_momentum_in,
            cb_scalar_args,
            cb_tmp4,
            /*idxA=*/0,
            /*idxB=*/momentum_tile,
            /*popA=*/true,
            /*popB=*/false>();

        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Add,
            cb_tmp3,
            cb_tmp4,
            cb_tmp1,
            /*idxA=*/0,
            /*idxB=*/0,
            /*popA=*/true,
            /*popB=*/true>();

        cb_momentum_tmp = cb_tmp1;
#endif

        copy_tile_to_cb(cb_momentum_tmp, cb_momentum_out, 0, /*pop=*/0);

#if defined(NESTEROV)
        // grad = grad + momentum_v * momentum
        uint32_t pop_momentum = (cb_grad_tmp != cb_momentum_tmp);
        mul_tiles_to_cb(cb_momentum_tmp, cb_scalar_args, cb_tmp3, 0, momentum_tile, /*pop0=*/pop_momentum, /*pop1=*/0);

        add_tiles_to_cb(cb_tmp3, cb_grad_tmp, cb_tmp4, 0, 0, /*pop0=*/1, /*pop1=*/1);

        cb_grad_tmp = cb_tmp4;
#else
// have to pop cb_grad_tmp
#if defined(MOMENTUM_INITIALIZED)
        cb_pop_front(cb_grad_tmp, 1);
#else
// not pop this case because `cb_momentum_tmp == cb_grad_tmp`
#endif

        cb_grad_tmp = cb_momentum_tmp;
#endif

#endif  // MOMENTUM

        // param_out = param_in - lr * grad
        mul_tiles_to_cb(cb_scalar_args, cb_grad_tmp, cb_tmp3, lr_tile, 0, /*pop0=*/0, /*pop1=*/1);

        moreh_bin_chain<
            compute_kernel_lib::BinaryFpuOp::Sub,
            cb_param_in,
            cb_tmp3,
            cb_param_out,
            /*idxA=*/0,
            /*idxB=*/0,
            /*popA=*/true,
            /*popB=*/true>();
    }
}
