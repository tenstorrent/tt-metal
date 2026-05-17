// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

namespace {

// PARTIAL migration helper — chains the three Mul stages of moreh_norm_backward.
// The bcast dimension is selected at compile time from the (ht_need_bcast,
// wt_need_bcast) flags. Each call: WaitAndPop on the A operand (cb_a is freshly
// pushed and popped by this stage), NoWaitNoPop on B (caller controls B
// lifecycle), per-tile reserve+push on cb_out.
template <
    bool HtBcast,
    bool WtBcast,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    compute_kernel_lib::CopyTilePolicy APolicy = compute_kernel_lib::CopyTilePolicy::WaitAndPop,
    compute_kernel_lib::CopyTilePolicy BPolicy = compute_kernel_lib::CopyTilePolicy::NoWaitNoPop>
ALWI void moreh_norm_bwd_mul_chain() {
    using namespace compute_kernel_lib;
    constexpr BroadcastDim Bcast = (HtBcast && WtBcast) ? BroadcastDim::Scalar
                                   : HtBcast            ? BroadcastDim::Row
                                   : WtBcast            ? BroadcastDim::Col
                                                        : BroadcastDim::None;
    eltwise_chain(
        1,
        BinaryFpu<
            CbA,
            CbB,
            BinaryFpuOp::Mul,
            Bcast,
            BinaryDataFormatReconfig::Input,
            APolicy,
            BPolicy,
            CbIndexMode::FirstTile,
            Dst::D0>{},
        PackTile<
            CbOut,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::Output>{});
}

}  // namespace

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    // runtime args
    int i{0};
    const auto num_input_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto p_minus_one = get_arg_val<uint32_t>(i++);
    const bool p_minus_one_is_negative = get_arg_val<uint32_t>(i++) == 1;

    // CB ids made constexpr to allow chain template instantiation.
    constexpr uint32_t cb_x = tt::CBIndex::c_0;        // input(==x)
    constexpr uint32_t cb_y = tt::CBIndex::c_1;        // output(==y)
    constexpr uint32_t cb_dy = tt::CBIndex::c_2;       // output_grad(==dy)
    constexpr uint32_t cb_decimal = tt::CBIndex::c_3;  // decimal

    constexpr uint32_t cb_dx = tt::CBIndex::c_16;  // input_grad(==dx)

    constexpr uint32_t cb_tmp0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_27;
    constexpr uint32_t cb_tmp4 = tt::CBIndex::c_28;
    constexpr uint32_t cb_tmp5 = tt::CBIndex::c_29;
    constexpr uint32_t cb_tmp6 = tt::CBIndex::c_30;
    constexpr uint32_t cb_tmp7 = tt::CBIndex::c_31;

    constexpr uint32_t cb_xpow = cb_tmp0;
    constexpr uint32_t cb_logx = cb_tmp1;
    constexpr uint32_t cb_exp_lxmd = cb_tmp2;
    constexpr uint32_t cb_correct_xpow = cb_tmp3;
    constexpr uint32_t cb_recip_ypow = cb_tmp6;
    constexpr uint32_t cb_sign = cb_tmp7;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);
    cb_wait_front(cb_decimal, onetile);  // comes from the reader

    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        cb_wait_front(cb_x, onetile);   // comes from the reader
        cb_wait_front(cb_y, onetile);   // comes from the reader
        cb_wait_front(cb_dy, onetile);  // comes from the reader

        sign_tile_to_cb(cb_x, cb_sign, 0, /*pop=*/0);

        // x^(p - 1)
        power_tile_with_abs_x_to_cb(
            cb_x, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow, p_minus_one, p_minus_one_is_negative);

        // x^(p - 1) * y -> cb_tmp4
        // Note: cb_y wait is the outer-loop wait (cb_y stays alive across the
        // three multiplies + the 1/y^p helper). Caller pops cb_y at the end.
        moreh_norm_bwd_mul_chain<ht_need_bcast, wt_need_bcast, cb_correct_xpow, cb_y, cb_tmp4>();

        // x^(p - 1) * y * dy -> cb_tmp5
        moreh_norm_bwd_mul_chain<ht_need_bcast, wt_need_bcast, cb_tmp4, cb_dy, cb_tmp5>();

        // 1 / y^p
        power_and_recip_tile_to_cb(cb_y, cb_tmp0, cb_tmp1, cb_decimal, cb_tmp2, cb_recip_ypow, p, p_is_negative);

        // (x^(p - 1) * y * dy) / y^p -> cb_tmp4
        moreh_norm_bwd_mul_chain<
            ht_need_bcast,
            wt_need_bcast,
            cb_tmp5,
            cb_recip_ypow,
            cb_tmp4,
            compute_kernel_lib::CopyTilePolicy::WaitAndPop,
            compute_kernel_lib::CopyTilePolicy::WaitAndPop>();

        cb_pop_front(cb_dy, onetile);

        // multiply abs sign
        mul_tiles_to_cb(cb_sign, cb_tmp4, cb_dx, 0, 0);
    }

    cb_pop_front(cb_decimal, onetile);
}
