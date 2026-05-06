// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto recip_p = get_arg_val<uint32_t>(i++);
    const bool recip_p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr std::uint32_t cb_x = tt::CBIndex::c_0;                // input
    constexpr std::uint32_t cb_one = tt::CBIndex::c_1;              // one
    constexpr std::uint32_t cb_decimal = tt::CBIndex::c_2;          // decimal
    constexpr std::uint32_t cb_recip_p_decimal = tt::CBIndex::c_3;  // recip_p_decimal
    constexpr std::uint32_t cb_mask_h = tt::CBIndex::c_4;           // mask_h

    constexpr std::uint32_t cb_y = tt::CBIndex::c_16;  // output

    constexpr std::uint32_t cb_tmp0 = tt::CBIndex::c_24;
    constexpr std::uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr std::uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr std::uint32_t cb_tmp3 = tt::CBIndex::c_27;
    constexpr std::uint32_t cb_tmp4 = tt::CBIndex::c_28;
    constexpr std::uint32_t cb_tmp5 = tt::CBIndex::c_29;
    constexpr std::uint32_t cb_tmp6 = tt::CBIndex::c_30;

    constexpr std::uint32_t cb_xabs = cb_tmp0;          // |x|
    constexpr std::uint32_t cb_xpow = cb_tmp1;          // |x|^p
    constexpr std::uint32_t cb_logx = cb_tmp2;          // log(|x|)
    constexpr std::uint32_t cb_exp_lxmd = cb_tmp3;      // exp(log(|x|) * decimal)
    constexpr std::uint32_t cb_correct_xpow = cb_tmp4;  // |x|^p * exp(log(|x|) * decimal)(==|x + decimal|^p)
    constexpr std::uint32_t cb_xpowadd = cb_tmp5;       // Add(|x + decimal|^p)
    constexpr std::uint32_t cb_xpowsum = cb_tmp6;       // Sum(|x + decimal|^p)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    cb_wait_front(cb_one, onetile);              // comes from the reader
    cb_wait_front(cb_decimal, onetile);          // comes from the reader
    cb_wait_front(cb_recip_p_decimal, onetile);  // comes from the reader

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    if (do_mask_h) {
        cb_wait_front(cb_mask_h, onetile);  // comes from the reader
    }

    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            // |x|
            tile_regs_acquire();
            cb_wait_front(cb_x, onetile);  // comes from the reader
            cb_reserve_back(cb_xabs, onetile);

            copy_tile_init_with_dt(cb_x);
            copy_tile(cb_x, 0, dst0);

            if (do_mask_h && (row_idx == Ht - 1)) {
                copy_tile_init_with_dt(cb_mask_h);
                copy_tile(cb_mask_h, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            abs_tile_init();
            abs_tile(dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_xabs);
            tile_regs_release();

            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_xabs, onetile);

            power_tile_to_cb(cb_xabs, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow, p, p_is_negative);

            // Add(|x|^p)
            if (row_idx == 0) {
                // PARTIAL migration: seed cb_xpowadd with first cb_correct_xpow tile.
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_correct_xpow);
                pack_reconfig_data_format(cb_xpowadd);
#endif
                {
                    using namespace compute_kernel_lib;
                    eltwise_chain(
                        onetile,
                        CopyTile<cb_correct_xpow, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                        PackTile<cb_xpowadd, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
                }
            } else {
                tile_regs_acquire();
                cb_wait_front(cb_correct_xpow, onetile);
                cb_wait_front(cb_xpowadd, onetile);
                cb_reserve_back(cb_xpowadd, onetile);

                add_tiles_init_with_dt(cb_correct_xpow, cb_xpowadd);
                add_tiles(cb_correct_xpow, cb_xpowadd, 0, 0, dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_xpowadd);
                tile_regs_release();

                cb_pop_front(cb_correct_xpow, onetile);
                cb_pop_front(cb_xpowadd, onetile);
                cb_push_back(cb_xpowadd, onetile);
            }
        }
        // Sum(|x|^p) - reduce single pre-accumulated tile
        compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
            cb_xpowadd, cb_one, cb_xpowsum, compute_kernel_lib::ReduceInputBlockShape::single());

        power_tile_to_cb(cb_xpowsum, cb_tmp0, cb_tmp1, cb_recip_p_decimal, cb_tmp2, cb_y, recip_p, recip_p_is_negative);
    }

    cb_pop_front(cb_one, onetile);
    cb_pop_front(cb_decimal, onetile);
    cb_pop_front(cb_recip_p_decimal, onetile);
    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
}
