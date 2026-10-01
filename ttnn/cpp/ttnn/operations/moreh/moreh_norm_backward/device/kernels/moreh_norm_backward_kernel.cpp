// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"       // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"       // Abs, Sign
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

void kernel_main() {
    // compile-time args
    constexpr bool wt_need_bcast = (get_arg(args::wt_need_bcast) == 1);
    constexpr bool ht_need_bcast = (get_arg(args::ht_need_bcast) == 1);

    constexpr auto kBcast = (ht_need_bcast && wt_need_bcast) ? ckl::BroadcastDim::Scalar
                            : ht_need_bcast                  ? ckl::BroadcastDim::Row
                            : wt_need_bcast                  ? ckl::BroadcastDim::Col
                                                             : ckl::BroadcastDim::None;

    // runtime args
    const auto num_input_tiles_per_core = get_arg(args::num_input_tiles_per_core);
    const auto p = get_arg(args::p);
    const bool p_is_negative = get_arg(args::p_is_negative) == 1;
    const auto p_minus_one = get_arg(args::p_minus_one);
    const bool p_minus_one_is_negative = get_arg(args::p_minus_one_is_negative) == 1;

    DataflowBuffer dfb_x_obj(dfb::x);              // input(==x), c_0
    DataflowBuffer dfb_y_obj(dfb::y);              // output(==y), c_1
    DataflowBuffer dfb_dy_obj(dfb::dy);            // output_grad(==dy), c_2
    DataflowBuffer dfb_decimal_obj(dfb::decimal);  // decimal, c_3

#ifdef NORM_INF
    // Only the +/-inf sub-gradient path still drives these buffers by hand.
    DataflowBuffer dfb_dx_obj(dfb::dx);
    DataflowBuffer dfb_tmp4_obj(dfb::tmp4);
    DataflowBuffer dfb_tmp5_obj(dfb::tmp5);
    DataflowBuffer dfb_sign_obj(dfb::sign);
    constexpr uint32_t dst0 = 0;
#endif

    constexpr uint32_t onetile = 1;

    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::dx);
    dfb_decimal_obj.wait_front(onetile);  // comes from the reader

    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        dfb_x_obj.wait_front(onetile);   // comes from the reader
        dfb_y_obj.wait_front(onetile);   // comes from the reader
        dfb_dy_obj.wait_front(onetile);  // comes from the reader

        sign_tile_to_dfb<dfb::x, dfb::sign>(0, /*pop=*/0);

#ifdef NORM_INF
        // ±inf sub-gradient: dx = sign(x) * dy * eq(|x|, y). The mask eq(|x| - y, 0) selects the
        // argmax(|x|) set for p = +inf and the argmin(|x|) set for p = -inf, because y equals
        // the norm value (max|x| / min|x|) either way — only equality matters, not the order.
        // step 1: tmp4 = |x|
        {
            dfb_tmp4_obj.reserve_back(onetile);
            tile_regs_acquire();
            copy_tile_init_with_dt(dfb_x_obj);
            copy_tile(dfb::x, 0, dst0);
            abs_tile_init();
            abs_tile(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_tmp4_obj);
            tile_regs_release();
            dfb_tmp4_obj.push_back(onetile);
        }
        // step 2: tmp5 = |x| - y   (y broadcast along the reduced dims; A = full tile, B = y)
        {
            dfb_tmp4_obj.wait_front(onetile);
            dfb_tmp5_obj.reserve_back(onetile);
            tile_regs_acquire();
            if (ht_need_bcast && wt_need_bcast) {
                sub_bcast_scalar_init_with_dt(dfb_tmp4_obj, dfb_y_obj);
                sub_tiles_bcast_scalar(dfb::tmp4, dfb::y, 0, 0, dst0);
            } else if (ht_need_bcast) {
                sub_bcast_rows_init_with_dt(dfb_tmp4_obj, dfb_y_obj);
                sub_tiles_bcast_rows(dfb::tmp4, dfb::y, 0, 0, dst0);
            } else if (wt_need_bcast) {
                sub_bcast_cols_init_with_dt(dfb_tmp4_obj, dfb_y_obj);
                sub_tiles_bcast_cols(dfb::tmp4, dfb::y, 0, 0, dst0);
            } else {
                sub_tiles_init_with_dt(dfb_tmp4_obj, dfb_y_obj);
                sub_tiles(dfb::tmp4, dfb::y, 0, 0, dst0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_tmp5_obj);
            tile_regs_release();
            dfb_tmp4_obj.pop_front(onetile);
            dfb_tmp5_obj.push_back(onetile);
        }
        // step 3: tmp4 = eq(tmp5, 0) — the argmax(|x|) / argmin(|x|) mask
        {
            dfb_tmp5_obj.wait_front(onetile);
            dfb_tmp4_obj.reserve_back(onetile);
            tile_regs_acquire();
            copy_tile_init_with_dt(dfb_tmp5_obj);
            copy_tile(dfb::tmp5, 0, dst0);
            unary_eq_tile_init();
            unary_eq_tile(dst0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_tmp4_obj);
            tile_regs_release();
            dfb_tmp5_obj.pop_front(onetile);
            dfb_tmp4_obj.push_back(onetile);
        }
        // step 4: tmp5 = sign(x) * dy  (A = sign full tile, B = dy broadcast along reduced dims)
        {
            dfb_sign_obj.wait_front(onetile);
            dfb_tmp5_obj.reserve_back(onetile);
            tile_regs_acquire();
            if (ht_need_bcast && wt_need_bcast) {
                mul_bcast_scalar_init_with_dt(dfb_sign_obj, dfb_dy_obj);
                mul_tiles_bcast_scalar(dfb::sign, dfb::dy, 0, 0, dst0);
            } else if (ht_need_bcast) {
                mul_bcast_rows_init_with_dt(dfb_sign_obj, dfb_dy_obj);
                mul_tiles_bcast_rows(dfb::sign, dfb::dy, 0, 0, dst0);
            } else if (wt_need_bcast) {
                mul_bcast_cols_init_with_dt(dfb_sign_obj, dfb_dy_obj);
                mul_tiles_bcast_cols(dfb::sign, dfb::dy, 0, 0, dst0);
            } else {
                mul_tiles_init_with_dt(dfb_sign_obj, dfb_dy_obj);
                mul_tiles(dfb::sign, dfb::dy, 0, 0, dst0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_tmp5_obj);
            tile_regs_release();
            dfb_sign_obj.pop_front(onetile);
            dfb_tmp5_obj.push_back(onetile);
        }
        // step 5: dx = (sign(x) * dy) * mask
        {
            dfb_tmp5_obj.wait_front(onetile);
            dfb_tmp4_obj.wait_front(onetile);
            dfb_dx_obj.reserve_back(onetile);
            tile_regs_acquire();
            mul_tiles_init_with_dt(dfb_tmp5_obj, dfb_tmp4_obj);
            mul_tiles(dfb::tmp5, dfb::tmp4, 0, 0, dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_dx_obj);
            tile_regs_release();
            dfb_tmp5_obj.pop_front(onetile);
            dfb_tmp4_obj.pop_front(onetile);
            dfb_dx_obj.push_back(onetile);
        }

        dfb_x_obj.pop_front(onetile);
        dfb_y_obj.pop_front(onetile);
        dfb_dy_obj.pop_front(onetile);
#else
        // x^(p - 1)
        power_tile_with_abs_x_to_dfb<dfb::x, dfb::xpow, dfb::logx, dfb::decimal, dfb::exp_lxmd, dfb::correct_xpow>(
            p_minus_one, p_minus_one_is_negative);

        // x^(p - 1) * y -> dfb::tmp4
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb::correct_xpow, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::y,
                    kBcast,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset)>{},
            ckl::PackTile<ckl::output(
                dfb::tmp4, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // x^(p - 1) * y * dy -> dfb::tmp5
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb::tmp4, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::dy,
                    kBcast,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset)>{},
            ckl::PackTile<ckl::output(
                dfb::tmp5, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // 1 / y^p
        power_and_recip_tile_to_dfb<dfb::y, dfb::xpow, dfb::logx, dfb::decimal, dfb::exp_lxmd, dfb::recip_ypow>(
            p, p_is_negative);

        // (x^(p - 1) * y * dy) / y^p -> dfb::tmp4
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(dfb::tmp5, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::recip_ypow, kBcast, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
            ckl::PackTile<ckl::output(
                dfb::tmp4, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        dfb_dy_obj.pop_front(onetile);

        // multiply abs sign
        mul_tiles_to_dfb<dfb::sign, dfb::tmp4, dfb::dx>();
#endif
    }

    dfb_decimal_obj.pop_front(onetile);
}
